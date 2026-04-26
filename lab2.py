import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
load_dotenv()

api_key=os.getenv("OPENAI_API_KEY")
# print(api_key)

system_prompt="""
You are a professional chef

Rules:
- Always think step-by-step
- Analyze ingredients
- Suggest meals clearly
- Be friendly and human-like
"""

llm= ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=api_key
)
agent = create_agent(
    llm,
    system_prompt=system_prompt  
)
res =agent.invoke({
    "message":[
        HumanMessage(content="i have chicken and rice")
    ]
})

print(res)