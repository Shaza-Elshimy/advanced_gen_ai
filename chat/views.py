from django.shortcuts import render
from .ai import ask_chef


def home(request):
    result = None

    if request.method == "POST":
        ingredients = request.POST.get("ingredients") 
        result = ask_chef(ingredients)

    return render(request, "chat.html", {
        "result": result
    })