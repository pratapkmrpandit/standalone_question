from utils.crew.build import execute_crew
from agents.data_query_assistant import NLSQLAgent
from standalone import settings

class StandAloneAgent:
    def __init__(self):
        self.module_classifier = NLSQLAgent(
            llm_conf=settings.llm_conf,
            vec_db_conf=settings.vdb_conf,
            prompt_conf=settings.prompt_conf,
        )

    def get_standalone_question(self, user_query, history, prompt_conf=None, llm_conf=None):
        try:
            # Step 1: Generate standalone question using Crew AI
            inputs = {
                "question": user_query,
                "chat_history": history,
                "prompt_conf": prompt_conf or {"rephrase_ques_prompt": "Rephrase into a standalone question"},
                "llm_conf": llm_conf or {"model": "gemini/gemini-2.0-flash"}
            }
            crew_result = execute_crew(inputs)

            # Access the standalone question
            tasks_output = getattr(crew_result, "tasks_output", [])
            if tasks_output and hasattr(tasks_output[0], "raw"):
                standalone_question = tasks_output[0].raw
            else:
                standalone_question = user_query

                # Step 2: Classify module/component using the standalone question
            module_clf, open_ai_error_code, fewshot_sub_df, error, fewshot_fetch_time, module_classification_process_time, token_usage = \
                self.module_classifier.classify_module_component_name(
                    user_query=standalone_question,
                    node_name=[]  # or pass list of candidate modules if you have
                )

            # Return both standalone question and classification
            return {
                "standalone_question": standalone_question,
                "module_classification": module_clf,
                "error": error
            }

        except Exception as e:
            return {"error": str(e)}
