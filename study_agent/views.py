from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
from .rag import ask_rag

from django.shortcuts import render

def rag_page(request):
    if request.method == 'GET' and 'q' in request.GET:
        question = request.GET.get("q")

        if not question:
            return JsonResponse({"error": "No question provided"})

        # get history
        history = request.session.get("history", [])

        answer = ask_rag(question)

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
    return render(request, "study_agent/rag.html")