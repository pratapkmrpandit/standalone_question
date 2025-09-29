import os
import json
import pandas as pd
import requests
from dotenv import load_dotenv
from openai import OpenAI

class ModuleClassifier:
    def __init__(self):
        # Load environment variables from .env
        load_dotenv()

        # Set Azure OpenAI environment variables
        os.environ["OPENAI_API_TYPE"] = "azure"
        os.environ["OPENAI_API_BASE"] = os.getenv("AZURE_OPENAI_ENDPOINT")
        os.environ["OPENAI_API_VERSION"] = os.getenv("AZURE_OPENAI_API_VERSION")

        # Initialize Azure OpenAI client
        self.client = OpenAI(api_key=os.getenv("AZURE_OPENAI_API_KEY"))

        # Azure Cognitive Search settings
        self.search_endpoint = os.getenv("AZURE_AI_SEARCH_ENDPOINT")
        self.search_api_key = os.getenv("AZURE_AI_SEARCH_API_KEY")
        self.component_index = os.getenv("AZURE_AI_SEARCH_COMPONENT_INDEX")

        # Models
        self.embedding_model = "text-embedding-3-small"
        self.chat_model = "gpt-4-mini"

    def get_embedding(self, text: str):
        """Generate embedding for a given text using Azure OpenAI."""
        try:
            response = self.client.embeddings.create(model=self.embedding_model, input=text)
            return response.data[0].embedding
        except Exception as e:
            raise RuntimeError(f"Error generating embedding: {str(e)}")

    def search_similar_docs(self, query_vector, top_k=5):
        """Search Azure Cognitive Search using vector search."""
        try:
            url = f"{self.search_endpoint}/indexes/{self.component_index}/docs/search?api-version=2023-07-01-Preview"
            headers = {
                "Content-Type": "application/json",
                "api-key": self.search_api_key
            }
            body = {
                "vector": {"value": query_vector, "fields": "contentVector", "k": top_k},
                "select": "id,modulename,componentname,tablename,sqlquery,page_content"
            }
            response = requests.post(url, headers=headers, json=body)
            response.raise_for_status()
            return response.json().get("value", [])
        except Exception as e:
            raise RuntimeError(f"Error searching Cognitive Search: {str(e)}")

    def prepare_fewshot_df(self, docs, user_query):
        """Convert search results to pandas DataFrame for LLM few-shot examples."""
        fewshot_li = []
        for doc in docs:
            fewshot_li.append([
                doc.get('id'),
                user_query,
                doc.get('page_content', ''),
                doc.get('modulename', 'NA'),
                doc.get('componentname', 'NA'),
                doc.get('tablename', 'NA'),
                doc.get('sqlquery', 'NA')
            ])
        return pd.DataFrame(
            fewshot_li,
            columns=["ID", "UserQuery", "FewShot", "Module", "Component", "TableList", "SQLQuery"]
        )

    def classify_module_component_name(self, user_query: str, node_name=[]):
        """Full pipeline: embedding -> search -> few-shot -> LLM classification."""
        try:
            # Step 1: Generate embedding
            print("hiiii")
            embedding = self.get_embedding(user_query)

            # Step 2: Search top-k similar docs
            docs = self.search_similar_docs(embedding, top_k=5)

            # Step 3: Prepare few-shot DataFrame
            fewshot_df = self.prepare_fewshot_df(docs, user_query)

            # Step 4: LLM classification using few-shot context
            messages = [
                {"role": "system", "content": "You are a module/component classification assistant."},
                {"role": "assistant", "content": str(fewshot_df[['FewShot','ID']].values)},
                {"role": "user", "content": user_query}
            ]

            llm_response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=messages
            )
            llm_out = llm_response.choices[0].message.content

            # Step 5: Parse JSON output
            try:
                module_clf = json.loads(llm_out)
            except:
                module_clf = {
                    "nodename": "NA",
                    "componentname": "NA",
                    "tableslist": "NA",
                    "followupquery": None
                }

            # Optional metadata
            open_ai_error_code = None
            fewshot_sub_df = fewshot_df
            error = None
            fewshot_fetch_time = 0
            module_classification_process_time = 0
            token_usage = {}

            return module_clf, open_ai_error_code, fewshot_sub_df, error, fewshot_fetch_time, module_classification_process_time, token_usage

        except Exception as e:
            return {"nodename": "NA"}, str(e), pd.DataFrame(), str(e), 0, 0, {}
