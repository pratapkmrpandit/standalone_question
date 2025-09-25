from utils.crew.build import execute_crew

class StandAloneAgent:
    def get_standalone_question(self, user_query, history, prompt_conf=None, llm_conf=None):
        try:
            inputs = {
                "question": user_query,
                "chat_history": history,
                "prompt_conf": prompt_conf or {"rephrase_ques_prompt": "Rephrase into a standalone question"},
                "llm_conf": llm_conf or {"model": "gemini/gemini-2.0-flash"}
            }

            result = execute_crew(inputs)
            return result

        except Exception as e:
            return {"error": str(e)}
