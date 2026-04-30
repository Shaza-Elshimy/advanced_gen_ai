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
# print(vectorstore.similarity_search("what is primary key"))