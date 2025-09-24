import traceback
from utils.crew.build import execute_crew

class StandAloneAgent:

    def get_standalone_question(
        self,
        user_query,
        history,
        prompt_conf,
        llm_conf,
        history_context_length=5
    ):
        try:
            history_str = ""
            if history and len(history) > 0:
                # Take last N turns from history
                for ch in history[-history_context_length:]:
                    user_ques = ch.get("rephrased_question") or ch.get("user_question")
                    sys_resp = ch.get("genai_response")
                    status_code = ch.get("genai_response_status_code")
                    sql_generated = ch.get("sql_generated")

                    if status_code == 200:
                        history_str += (
                            f"User: {user_ques}\n"
                            f"System: {sys_resp}\n"
                            f"SQL Generated: {sql_generated}\n\n"
                        )

            # Prepare inputs for CrewAI
            inputs = {
                "user_query": user_query,
                "history": history_str,   # formatted history string
                "prompt_conf": prompt_conf,
                "llm_conf": llm_conf,
                "history_context_length": history_context_length,
            }

            result = execute_crew(inputs)
            return result

        except Exception as e:
            traceback.print_exc()
            return {
                "standalone_question": None,
                "error": str(e)
            }
