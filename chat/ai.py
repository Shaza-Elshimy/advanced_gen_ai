import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from typing import List

load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")
base_url="https://openrouter.ai/api/v1"
api_key = os.getenv("OPENROUTER_API_KEY")

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
    # model="gpt-4o-mini",
    model="poolside/laguna-xs.2:free",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url
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
# ------------------------------------------------------------
import base64

vision_llm = ChatOpenAI(
    # model="gpt-4o-mini",
    model="poolside/laguna-xs.2:free",
    temperature=0,
    api_key=api_key,
    base_url=base_url
)

def extract_ingredients_from_image(image_file):

    image_bytes = image_file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": "Extract all food ingredients as a comma-separated list."
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
    )

    response = vision_llm.invoke([message])
    return response.content