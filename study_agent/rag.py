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

question="what is primary key?"

docs = retriever.invoke(question)
context ="\n".join([doc.page_content for doc in docs])

final_prompt = rag_prompt.format(
    context=context,
    question=question
)

response = llm.invoke(final_prompt)

print(response.content)
