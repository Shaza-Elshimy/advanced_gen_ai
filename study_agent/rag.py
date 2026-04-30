import pathlib 
from langchain_community.document_loaders import PyPDFLoader

paths =list(pathlib.Path("./study_agent/data").glob("**/*.pdf"))
print(paths)

documents=[]
for path in paths:
    loader =PyPDFLoader(str(path))
    docs =loader.load()
    documents.extend(docs)
print(len(documents))