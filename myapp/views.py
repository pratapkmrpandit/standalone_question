import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from agents.stand_alone import StandAloneAgent

@csrf_exempt
def standalone_question_view(request):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)

    try:
        data = json.loads(request.body)
        user_query = data.get("question", "")
        history = data.get("chat_history", [])

        agent = StandAloneAgent()
        result = agent.get_standalone_question(user_query=user_query, history=history)

        return JsonResponse(result, safe=True)

    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
