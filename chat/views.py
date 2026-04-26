from django.shortcuts import render
from .ai import ask_chef,chat_history
# Create your views here.

def home(request):
    response = ""

    if request.method == "POST":
        ingredients = request.POST.get("ingredients")
        response = ask_chef(ingredients)

    return render(request, "chat.html", {"response": response,"history":chat_history})