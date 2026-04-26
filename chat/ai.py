import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.7
)

chat_history = []

def ask_chef(user_input):
    global chat_history

    chat_history.append(HumanMessage(content=user_input))

    response = llm.invoke(chat_history)

    chat_history.append(AIMessage(content=response.content))

    return response.content