import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


system_prompt = """
You are a Nutrition AI Assistant.

- Analyze meals from text or image
- Estimate calories and nutrients
- Suggest improvements

Always include:
"This is not medical or dietary advice. Consult a qualified professional."
"""



llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=api_key
)

nutrition_agent = create_agent(
    llm,
    system_prompt=system_prompt,
    checkpointer=InMemorySaver()
)
