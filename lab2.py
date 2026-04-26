import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from pydantic import BaseModel
from typing import List
load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
# print(api_key)
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


system_prompt = """
You are a professional chef.

Rules:
- Always analyze ingredients step-by-step
- Never skip steps
- Suggest multiple meals
- Mark ingredients as available or not available
- Be human-like and friendly

IMPORTANT:
Return ONLY structured output in JSON format matching schema.
"""

structured_llm= ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=api_key
)

agent = create_agent(
    structured_llm,
    system_prompt=system_prompt  ,
    response_format=ChefResponse
)

res1 =agent.invoke({
    "message":[
        HumanMessage(content="i have chicken and rice give hoe can i cook with it")
    ]
    },
    config={"configurable":{
        "thread_id":"1"
    }
    }
    ) 
print("\n first respose: \n")
print(res1['messages'][-1].content)

res2=agent.invoke({
    "messages":[
        HumanMessage(content="give me another idea")
    ]
    },
    config={"configurable":{
        "thread_id":"1"
    }
    }
    ) 

print("\n second respose: \n")
print(res2['messages'][-1].content)