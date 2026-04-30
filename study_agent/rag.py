import pathlib 
from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter
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