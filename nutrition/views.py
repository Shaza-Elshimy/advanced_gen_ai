from django.shortcuts import render
from .agent import nutrition_agent
from langchain_core.messages import HumanMessage
import base64
def chat(request):

    if "nutrition_history" not in request.session:
        request.session["nutrition_history"] = []

    history = request.session["nutrition_history"]

    if request.method == "POST":

        meal = request.POST.get("meal")
        image = request.FILES.get("image")

        if image:
            import base64
            encoded = base64.b64encode(image.read()).decode("utf-8")

            content = [
                {"type": "text", "text": meal or "Analyze this meal"},
                {
                    "type": "image",
                    "base64": encoded,
                    "mime_type": "image/jpeg"
                }
            ]
        else:
            content = meal

        res = nutrition_agent.invoke(
            {
                "messages": [
                    HumanMessage(content=content)
                ]
            },
            config={"configurable": {"thread_id": "nutrition"}}
        )

        ai_response = res["messages"][-1].content


        history.append({
            "role": "user",
            "content": meal
        })

        history.append({
            "role": "ai",
            "content": ai_response
        })

        request.session["nutrition_history"] = history

    return render(request, "nutrition/chat.html", {
        "history": history
    })