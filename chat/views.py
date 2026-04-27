from django.shortcuts import render
from langchain_core.messages import HumanMessage
from .ai import ask_chef, extract_ingredients_from_image

def home(request):

    if "history" not in request.session:
        request.session["history"] = []

    history = request.session["history"]

    if request.method == "POST":

        user_input = request.POST.get("ingredients")
        image = request.FILES.get("image")

        if image:
            extracted = extract_ingredients_from_image(image)
            user_input = extracted

        messages = [HumanMessage(content=msg["content"]) for msg in history if msg["role"] == "user"]
        messages.append(HumanMessage(content=user_input))

        ai_response = ask_chef(messages)

        history.append({
            "role": "user",
            "content": user_input
        })

        history.append({
            "role": "assistant",
            "content": ai_response
        })

        request.session["history"] = history

    return render(request, "chat.html", {
        "history": history
    })