import os
import pathlib 
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
paths =list(pathlib.Path("./study_agent/data").glob("**/*.pdf"))
print(paths)

documents=[]
for path in paths:
    loader =PyPDFLoader(str(path))
    docs =loader.load()
    documents.extend(docs)
print(len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=100,
)
chunks = splitter.split_documents(documents)
print(len(chunks))

from langchain_openai import OpenAIEmbeddings

embedding =OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=api_key
)

from langchain_community.vectorstores import Chroma

vectorstore =Chroma.from_documents(
    collection_name="mysql_course",
    documents=chunks,
    embedding=embedding,
    persist_directory="chroma_db"
)
# print(vectorstore.similarity_search("what is primary key?"))

retriever =vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":3}
)

# docs = retriever.invoke("what is primary key?")

# print(docs)
# context ="\n".join([doc.page_content for doc in docs])

# print(context)

from langchain_openai import ChatOpenAI
llm=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

from langchain_core.prompts import PromptTemplate

rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a helpful assistant.
Answer ONLY using the context below.

If you don't find the answer, say "I don't know".

Context:
{context}

Question:
{question}
"""
)

# question="what is primary key?"

from langchain.tools import tool

@tool
def rag_tool(question: str):
    """
    Use this tool to answer questions about MySQL course PDFs.
    It retrieves relevant context from the vector database and answers based on it.
    """

    docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in docs])

    final_prompt = rag_prompt.format(
        context=context,
        question=question
    )

    response = llm.invoke(final_prompt)
    return response.content


from langchain_core.tools import tool
from tavily import TavilyClient
import os

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_KEY"))


@tool
def web_search(query: str) -> str:
    """Search the internet for any query using Tavily."""
    
    result = tavily_client.search(query)
    return str(result)

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver


from study_agent.models import Course

@tool
def search_courses(query: str) -> str:
    """
    Search for courses in the local database.
    """

    courses = Course.objects.filter(title__icontains=query)

    if not courses.exists():
        return "No courses found."

    result = []

    for course in courses:
        result.append(
            f"""
            Title: {course.title}
            Instructor: {course.instructor}
            Duration: {course.duration}
            Level: {course.level}
            """
        )

    return "\n".join(result)

system_prompt = """
YYou are a Study Assistant AI Agent.

You have access to these tools:

1. rag_tool
- Use it for questions about MySQL course PDFs.

2. web_search
- Use it for recent or online information.

3. search_courses
- Use it when the user asks about available courses.

Choose the correct tool automatically before answering.
"""

agent = create_agent(
    llm,
    system_prompt=system_prompt,
    tools=[rag_tool,web_search,search_courses],
    checkpointer=InMemorySaver()
)

result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="what is primary key?")
        ]
    },
    config={"configurable":{"thread_id":"1"}}
)

# print(result["messages"][-1].content)

