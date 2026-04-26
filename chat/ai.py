import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from typing import List


class Ingredient(BaseModel):
    name: str
    status: str  # available / not available


class Meal(BaseModel):
    name: str
    cooking_time: str
    number_of_individuals: int
    instructions: str
    ingredients: List[Ingredient]


class ChefResponse(BaseModel):
    meals: List[Meal]

load_dotenv()

system_prompt = """
You are a professional chef 👨‍🍳

Rules:
- Always analyze ingredients step-by-step
- Never skip steps
- Suggest multiple meals
- Mark ingredients as available or not available
- Be human-like and friendly

IMPORTANT:
Return ONLY structured output that matches the schema exactly.
"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

structured_llm = llm.with_structured_output(ChefResponse)


def ask_chef(user_input):
    full_input = system_prompt + "\nUser: " + user_input
    result = structured_llm.invoke(full_input)
    return result