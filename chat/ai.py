import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import List

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


class Ingredient(BaseModel):
    name: str
    status: str


class Meal(BaseModel):
    name: str
    cooking_time: str
    number_of_individuals: int
    instructions: str
    ingredients: List[Ingredient]


class ChefResponse(BaseModel):
    meals: List[Meal]


system_prompt = """
You are a professional chef 

Rules:
- Always analyze ingredients step-by-step
- Never skip steps
- Suggest multiple meals
- Mark ingredients as available or not available
- Be friendly and human-like

Return ONLY structured output matching schema.
"""


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=api_key
)

agent = create_agent(
    llm,
    system_prompt=system_prompt,
    response_format=ChefResponse
)


def ask_chef(messages):
    res = agent.invoke(
        {
            "messages": messages 
        },
        config={
            "configurable": {
                "thread_id": "chef_lab"
            }
        }
    )

    structured = res.get("structured_response")

    if structured:
        return structured.model_dump(), res["messages"][-1]

    return {"meals": []}