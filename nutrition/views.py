from django.shortcuts import render
# Create your views here.
from .agent import nutrition_agent
from langchain.messages import HumanMessage

def chat(request):

    response = None

    if request.method == "POST":
        meal = request.POST.get("meal")

        res = nutrition_agent.invoke(
            {
                "messages": [
                    HumanMessage(content=meal)
                ]
            },
            config={"configurable": {"thread_id": "nutrition"}}
        )

        response = res["messages"][-1].content

    return render(request, "nutrition/chat.html", {
        "response": response
    })