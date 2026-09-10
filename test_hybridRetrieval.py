# test_hybrid_retrieval.py
from ingestion import create_vectorstore
from retriever_builder import build_bm25_retriever, get_hybrid_retriever

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("document loader/CNN.pdf")
docs = loader.load()
vectorstore, _, _ = create_vectorstore(docs, "test_chroma_db")

# 🔍 DEBUG: Check what is actually stored in Chroma
data = vectorstore.get(include=["documents"])

print("\nTotal documents:", len(data["documents"]))

for i, doc in enumerate(data["documents"]):
    print(f"{i}: {doc[:80].replace(chr(10), ' ')}")
    
# Test BM25 alone
bm25 = build_bm25_retriever(vectorstore, k=5)
results = bm25.invoke("Pooling layer")   # exact term likely in the PDF
for r in results:
    print("BM25:", r.page_content[:100])

# Test hybrid
hybrid = get_hybrid_retriever(vectorstore, k=5)
results = hybrid.invoke("Pooling layer")
print(f"\nHybrid returned {len(results)} docs")
for r in results:
    print("HYBRID:", r.page_content[:100])