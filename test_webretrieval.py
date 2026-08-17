from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


# Same embedding model used during ingestion
embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)


# Load existing Chroma vector store
vectorstore = Chroma(
    persist_directory="test_web_vector_store",
    embedding_function=embedding_model
)


# Test question
question = "What services does The X Group provide?"


# Retrieve relevant chunks
results = vectorstore.similarity_search(
    question,
    k=4
)


print("\n==============================")
print("Retrieved documents:", len(results))
print("==============================")

for i, doc in enumerate(results, start=1):

    print(f"\n--- Result {i} ---")

    print("Source:", doc.metadata.get("source"))

    print("Title:", doc.metadata.get("title"))

    print("\nContent:")
    print(doc.page_content[:1000])