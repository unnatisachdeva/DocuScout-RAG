from ingestion import create_vectorstore
from retriever_builder import get_hybrid_retriever
from reranker import get_reranked_retriever

from langchain_community.document_loaders import PyPDFLoader


# -----------------------------
# Load PDF
# -----------------------------
loader = PyPDFLoader("document loader/CNN.pdf")
docs = loader.load()

vectorstore, _, _ = create_vectorstore(
    docs,
    "test_chroma_db"
)


# -----------------------------
# Hybrid Retriever
# -----------------------------
hybrid = get_hybrid_retriever(
    vectorstore,
    k=10
)


query = "What is max pooling and how does it work?"


# -----------------------------
# BEFORE RERANKING
# -----------------------------
hybrid_results = hybrid.invoke(query)

print("\n========== BEFORE RERANKING ==========\n")

for i, doc in enumerate(hybrid_results, 1):
    print(f"\n{i}.")
    print(doc.page_content[:300])


# -----------------------------
# RERANKER
# -----------------------------
reranked = get_reranked_retriever(
    hybrid,
    top_n=5
)


# -----------------------------
# AFTER RERANKING
# -----------------------------
reranked_results = reranked.invoke(query)

print("\n\n========== AFTER RERANKING ==========\n")

for i, doc in enumerate(reranked_results, 1):
    print(f"\n{i}.")
    print(doc.page_content[:300])