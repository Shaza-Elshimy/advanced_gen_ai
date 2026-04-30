from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from .rag import agent

from django.shortcuts import render

def rag_page(request):
    return render(request, "study_agent/rag.html", {'history': request.session.get('history', [])})
def rag_chat(request):
    question = request.GET.get("q")

    if not question:
        return JsonResponse({"error": "No question provided"})

    try:
        result = agent.invoke(
            {
                "messages": [
                    {"role": "user", "content": question}
                ]
            },
            config={
                "configurable": {
                    "thread_id": "django_user_1"
                }
            }
        )

        answer = result["messages"][-1].content
    except Exception as e:
        answer = f"Error: {str(e)}"

    # get history
    history = request.session.get("history", [])

    # add to history
    history.append({
        "question": question,
        "answer": answer
    })

    request.session["history"] = history

    return JsonResponse({
        "answer": answer,
        "history": history
    })