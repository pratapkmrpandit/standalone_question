import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from agents.stand_alone import StandAloneAgent

@csrf_exempt
def standalone_question_view(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)

            user_query = data.get("question", "")
            history = data.get("chat_history", [])
            user_details = data.get("user_details", {})

            prompt_conf = {"rephrase_ques_prompt": "Rephrase the question into a standalone form"}
            llm_conf = {"model": "gemini/gemini-2.0-flash"}

            agent = StandAloneAgent()
            result = agent.get_standalone_question(
                user_query=user_query,
                history=history,
                prompt_conf=prompt_conf,
                llm_conf=llm_conf
            )

            try:
                result_dict = result.dict()  # or use vars(result) if no dict() method
            except AttributeError:
                result_dict = vars(result)

            return JsonResponse(result_dict, safe=False)


        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Only POST method allowed"}, status=405)
