'''ingestion.py creates the vectorstore '''
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


def create_vectorstore(pdf_path, persist_directory):

    # Load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    # Embedding model
    embedding_model = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001"
    )

    # Create vector database
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )

    return vectorstore, len(docs), len(chunks)