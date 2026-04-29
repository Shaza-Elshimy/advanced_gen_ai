import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from tavily import TavilyClient
from langchain.tools import tool
import csv
load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")
base_url="https://openrouter.ai/api/v1"
api_key = os.getenv("OPENROUTER_API_KEY")
tavily_key=os.getenv("TAVILY_KEY")

system_prompt = """
You are a Nutrition AI Assistant.

You must:
- Analyze food (text or image)
- Estimate calories
- Suggest healthy alternatives
- Use tools when needed:
  - search healthy places
  - store analysis in CSV

Always include disclaimer:
"This is not medical or dietary advice.
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
llm = ChatOpenAI(
    model="poolside/laguna-xs.2:free",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url
)

nutrition_agent = create_agent(
    llm,
    system_prompt=system_prompt,
    tools=[search_healthy_places,store_nutrition],
    checkpointer=InMemorySaver()
)
