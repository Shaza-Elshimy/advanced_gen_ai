import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from langchain.agents.middleware import before_agent
from langgraph.runtime import Runtime
from langchain.agents import AgentState

from langchain_core.messages import (
    ToolMessage,
    AIMessage,
    RemoveMessage)

from tavily import TavilyClient
from langchain.tools import tool

from pydantic import BaseModel
from typing import List

import csv
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
# base_url="https://openrouter.ai/api/v1"
# api_key = os.getenv("OPENROUTER_API_KEY")
tavily_key=os.getenv("TAVILY_KEY")

class MealAnalysis(BaseModel):
    meal: str
    calories: str
    summary: str
    recommendations: List[str]
    disclaimer: str

system_prompt = """
You are a Nutrition AI Assistant.

You MUST:
1. Analyze food input (text or image-derived text)
2. Estimate calories and nutrition
3. Decide when to use tools:
   - Use search tool for healthy restaurants or nutrition info
   - Use storage tool to save meals and analysis
4. Provide clear structured response:
   - Meal Analysis
   - Calories
   - Summary
   - Recommendations

IMPORTANT:
- You may call multiple tools in sequence
- Always be structured and clear
- Never skip tool usage if needed

Always include this disclaimer:
"This is not medical or dietary advice. Consult a qualified professional."
"""

@tool
def search_healthy_places(meal: str, city: str) -> str:
    """Search for healthy restaurants based on meal and city"""
    client = TavilyClient(api_key=tavily_key)

    query = f"healthy restaurants or meals for {meal} in {city}"
    result = client.search(query)

    return str(result)



@tool
def store_nutrition(meal: str, calories: str, summary: str) -> str:
    """Store nutrition analysis in CSV"""

    with open("nutrition_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([meal, calories, summary])

    return "Saved successfully"



@before_agent
def trim_messages(
    state: AgentState,
    runtime: Runtime
) -> AgentState:

    """Remove tool messages and empty messages"""

    messages_to_remove = [
        msg
        for msg in state["messages"]
        if (
            isinstance(msg, ToolMessage)
            or str(msg.content).strip() == ""
        )
    ]

    return {
        "messages": [
            RemoveMessage(id=msg.id)
            for msg in messages_to_remove
        ]
    }

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=api_key,
    # base_url=base_url
)

nutrition_agent = create_agent(
    llm,
    system_prompt=system_prompt,
    tools=[search_healthy_places,store_nutrition],
    checkpointer=InMemorySaver(),
    response_format=MealAnalysis,
    middleware=[trim_messages],
)
