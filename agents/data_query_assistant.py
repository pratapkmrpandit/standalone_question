import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json
import os
import logging
import traceback
from langchain_openai import AzureOpenAIEmbeddings
from agents.azureSearch import AzureSearchAgent
from agents.open_ai import Openai
from standalone import settings
from dotenv import load_dotenv
load_dotenv()


class NLSQLAgent:
    def __init__(self, llm_conf, vec_db_conf, prompt_conf):
        self.logger = logging.getLogger(__name__)

        self.azure_agent = AzureSearchAgent()
        self.open_ai_error_code = None
        self.llm_conf = llm_conf
        self.vec_db_conf = vec_db_conf
        self.prompt_conf = prompt_conf
        self.index_name = vec_db_conf["collection_name"]
        self.component_index = vec_db_conf["component_index"]
        self.embeddings = AzureOpenAIEmbeddings(
            model=settings.config['embedding_model'],
            api_version=os.getenv("EMBEDDING_API_VERSION"),
        )


        self.component_index_vector_store = None
        self.vector_store = None

        try:
            if self.component_index:
                self.component_index_vector_store, self.open_ai_error_code = (
                    self.azure_agent.get_component_index()
                )
        except Exception as e:
            open_ai_error_code = "generic_error"
            self.logger.error(f"Error initializing component index {self.component_index}: {e}")
            traceback.print_exc()

        if self.index_name:
            self.vector_store, self.open_ai_error_code = (
                self.azure_agent.get_azure_index()
            )
        self.llm = None

        self.openai_error = "Error from OpenAI"
        if llm_conf["llm_model"] == "openai":
            self.llm = Openai()


    def classify_module_component_name(self, user_query, node_name):
        module_clf = {
            "nodename": "NA",
            "componentname": "NA",
            "tableslist": "NA",
            "followupquery": None,
        }
        open_ai_error_code = None
        fewshot_sub_df = pd.DataFrame()
        error = None
        module_classification_process_time = None
        fewshot_fetch_time = None
        tokens_info = {}
        token_usage = {}
        self.logger.info("CAME to classify_module_component_name")
        try:
            if not self.component_index_vector_store:
                self.component_index_vector_store, open_ai_error_code = (
                    self.azure_agent.get_component_index()
                )
            start_time = time.time()
            identified_modules = settings.config.get('modules_list', [])

            if any(mod in identified_modules for mod in node_name):
                # Build all possible module name variants
                node_name_stripped = [mod.strip() for mod in node_name]
                combined_modulename = ", ".join(node_name_stripped)
                combined_modulename_nospace = ",".join(node_name_stripped)
                filter_query_mod = " or ".join(
                    [f"modulename eq '{mod}'" for mod in node_name_stripped] +
                    [f"modulename eq '{combined_modulename}'"] +
                    [f"modulename eq '{combined_modulename_nospace}'"]
                )
                fewshot_str = f"source eq 'fewshot' and ({filter_query_mod})"
                comp_str = f"source eq 'doc' and ({filter_query_mod})"
            else:
                fewshot_str = "source eq 'fewshot'"
                comp_str = "source eq 'doc'"
            if self.component_index_vector_store is None:
                self.logger.error(
                    f"Component index '{self.component_index}' not initialized. Skipping similarity search."
                )
            else:
                with ThreadPoolExecutor() as executor:
                    future_docs = executor.submit(
                        self.component_index_vector_store.similarity_search, query=user_query, k=5, filters=comp_str
                    )
                    future_fewshots = executor.submit(
                        self.component_index_vector_store.similarity_search, query=user_query, k=5, filters=fewshot_str
                    )

                    docs = future_docs.result()
                    fewshots = future_fewshots.result()
            end_time = time.time()
            fewshot_fetch_time = int(end_time - start_time)

            fewshot_li = []
            self.logger.info("FewShots:::::::::::::::::::::::::::::::")
            for fewshot in fewshots:
                self.logger.info(fewshot.page_content)

                temp_li = [
                    fewshot.metadata['id'],
                    user_query,
                    fewshot.page_content,
                    fewshot.metadata["modulename"],
                    fewshot.metadata["componentname"],
                    fewshot.metadata["tablename"],
                    fewshot.metadata["sqlquery"],
                ]

                fewshot_li.append(temp_li)

            fewshot_df = pd.DataFrame(
                fewshot_li,
                columns=[
                    "ID",
                    "UserQuery",
                    "FewShot",
                    "Module",
                    "Component",
                    "TableList",
                    "SQLQuery"
                ],
            )
            similarity_prompt = settings.prompt_conf.get("similarity_score_prompt", "similarity_score_prompt")

            messages = [{"role": "system", "content": similarity_prompt},
                        {"role": "assistant", "content": str(fewshot_df[['FewShot', 'ID']].values)},
                        {"role": "user", "content": user_query}]

            llm_out, open_ai_error_code, error, fewshot_classification_process_time, few_shot_tokens_info = self.llm.generate_openai_response(
                messages=messages, config=self.llm_conf, json_resp_flag=True
            )
            token_usage['similarity_score_llm'] = few_shot_tokens_info

            fewshot_score = pd.DataFrame(columns=['questions', 'score', 'ID'])
            if llm_out is not None:
                fewshot_score = json.loads(llm_out)
                fewshot_score = pd.DataFrame(fewshot_score.get("questions", []))

            fewshot_df = pd.merge(fewshot_df, fewshot_score, on='ID', how='left')
            fewshot_df = fewshot_df.sort_values(by="score", ascending=False)
            self.logger.info("Before :: Fetched few shots for module identification : ")
            self.logger.info(str(fewshot_df.shape))

            fewshot_sub_df = fewshot_df[
                fewshot_df["score"] >= settings.config["few_shot_th"]
                ]

            self.logger.info("After:: Fetched few shots for module identification : ")
            self.logger.info(str(fewshot_sub_df.shape))

            # fuzzratio = fewshot_df.iloc[0]["FuzzRatio"]
            if len(fewshot_sub_df) > 0 and fewshot_sub_df.iloc[0]["Module"] != "NA":
                nodename = fewshot_sub_df.iloc[0]["Module"]
                componentname = fewshot_sub_df.iloc[0]["Component"]
                tableslist = fewshot_sub_df.iloc[0]["TableList"]
                followupquery = fewshot_sub_df.iloc[0]["FewShot"]
                module_clf = {
                    "nodename": nodename,
                    "componentname": componentname,
                    "tableslist": tableslist,
                    "followupquery": followupquery,
                }

            else:
                doc_str = ""
                self.logger.info("Component Documents ::::::::::::::::::")
                for doc in docs:
                    self.logger.debug(doc.metadata)
                    doc_str += doc.page_content + "/n"

                messages = [
                    {
                        "role": "system",
                        "content": self.prompt_conf["node_comp_clf_prompt"],
                    },
                    {"role": "assistant",
                     "content": "Module summary : /n" + self.prompt_conf["component_imp_tab_summary"]},
                    {"role": "assistant", "content": "Similar components fetched : /n" + doc_str}
                ]

                fewshot_sub_df = fewshot_df[
                    fewshot_df["score"] >= settings.config["few_shot_th_2"]
                    ]
                self.logger.info("After second threshold:: Fetched few shots for module identification : ")
                self.logger.info(str(fewshot_sub_df.shape))

                if len(fewshot_sub_df) > 0:
                    messages.append(
                        {"role": "assistant", "content": "Fewshot examples : /n" + str(fewshot_sub_df.T.to_dict())})

                messages.append({"role": "user", "content": user_query})

                llm_out, open_ai_error_code, error, module_classification_process_time, tokens_info = self.llm.generate_openai_response(
                    messages=messages, config=self.llm_conf, json_resp_flag=True
                )
                module_clf = json.loads(llm_out)
                token_usage['component_name_identification_llm'] = tokens_info
                token_usage['fewshot_score_identification_llm'] = few_shot_tokens_info
        except Exception:
            self.logger.error(
                "something went wrong in classify_module_component_name ::"
                + traceback.format_exc()
            )
            error = traceback.format_exc()

        return module_clf, open_ai_error_code, fewshot_sub_df, error, fewshot_fetch_time, module_classification_process_time, token_usage
