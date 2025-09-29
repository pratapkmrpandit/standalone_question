import logging
import os
import traceback
import time
import openai
from openai import AzureOpenAI
from dotenv import load_dotenv
load_dotenv()


class Openai:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        )
        print(os.getenv("AZURE_OPENAI_API_KEY"))
        print(os.getenv("AZURE_OPENAI_ENDPOINT"))
        print(os.getenv("API_VERSION"))

    # @traceable  # Auto-trace this function
    def generate_openai_response(
            self, messages, config, streaming=0, json_resp_flag=False
    ):
        tokens_info = {}
        output = None
        open_ai_error_code = None
        error = None
        resp_time = None
        try:
            start_time = time.time()
            if json_resp_flag:
                response = self.client.chat.completions.create(
                    model=config["model"],
                    messages=messages,
                    temperature=0,
                    response_format={"type": "json_object"}

                )

                self.logger.info("openai resp object :: " + str(response))
                output = response.choices[0].message.content
                tokens_info['prompt_tokens'] = response.usage.prompt_tokens
                tokens_info['completion_tokens'] = response.usage.completion_tokens
                tokens_info['total_tokens'] = response.usage.total_tokens


            elif streaming == 1:
                output = self.client.chat.completions.create(
                    model=config["model"], messages=messages, temperature=0, stream=True
                )

            elif streaming == 0:
                response = self.client.chat.completions.create(
                    model=config["model"], temperature=0, messages=messages
                )

                self.logger.info("openai resp object :: " + str(response))

                output = response.choices[0].message.content
                tokens_info['prompt_tokens'] = response.usage.prompt_tokens
                tokens_info['completion_tokens'] = response.usage.completion_tokens
                tokens_info['total_tokens'] = response.usage.total_tokens

            end_time = time.time()
            resp_time = int(end_time - start_time)
            self.logger.info("output :: " + str(output))
        except openai.OpenAIError as e:
            error_text = str(e).lower()
            if "quota" in error_text:
                open_ai_error_code = "429_quota"
            elif "rate" in error_text:
                open_ai_error_code = "429_rate_limit"
            elif "timeout" in error_text or "apitimeout" in error_text:
                open_ai_error_code = "408"
            elif hasattr(e, "status_code"):
                # if the exception object actually has a status_code attribute
                open_ai_error_code = str(e.status_code)
            else:
                # default fallback
                open_ai_error_code = "500"

            self.logger.error(f"OpenAI error ({open_ai_error_code}): {e}")
        except Exception:
            error = str(traceback.format_exc())
            self.logger.error(
                "Error while calling Openai LLM : " + str(traceback.format_exc())
            )
        return output, open_ai_error_code, error, resp_time, tokens_info
