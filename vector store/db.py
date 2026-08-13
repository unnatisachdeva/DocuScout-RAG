from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

docs = [
    Document(page_content="CNN is a deep learning model."),
    Document(page_content="CNN is used for image classification."),
    Document(page_content="CNN works with images represented as arrays of pixels.")
]

# embedding model 
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

#create vector store object
vectorstore= Chroma.from_documents( #creates embeddings and stores them in a vector db
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma-db"
)

#retrieve data from vector store
result= vectorstore.similarity_search("what is CNN?", k=2) #retrieves the most relevant data from the vector db based on the query
for r in result :
    print(r.page_content)

retriever= vectorstore.as_retriever() #retrieves the most relevant data from the vector db based on the query
docs= retriever.invoke("what is CNN?", k=2) #retrieves the most relevant data from the vector db based on the query
for d in docs :
    print(d.page_content)