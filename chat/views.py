from django.shortcuts import render
from .ai import ask_chef


def home(request):

    if "history" not in request.session:
        request.session["history"] = []

    history = request.session["history"]

    if request.method == "POST":
        user_input = request.POST.get("ingredients")

        history.append({
            "role": "user",
            "content": user_input
        })


        ai_response = ask_chef(user_input)

        history.append({
            "role": "assistant",
            "content": ai_response
        })

        request.session["history"] = history

    return render(request, "chat.html", {
        "history": history
    })