'''retriever_builder.py builds retrieval infrastructure.

Input:
    A populated Chroma vectorstore

Output:
    A hybrid retriever combining dense (vector/MMR) and sparse (BM25) search
'''


from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def build_bm25_retriever(vectorstore, k=8):
    # Pull chunks straight back out of Chroma instead of persisting a
    # second copy on disk — BM25Retriever just needs the raw Documents.
    data = vectorstore.get(include=["documents", "metadatas"])

    docs = [
        Document(page_content=text, metadata=meta or {})
        for text, meta in zip(data["documents"], data["metadatas"])
    ]

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = k
    return bm25


def get_hybrid_retriever(vectorstore, k=8):
    vector_retriever = vectorstore.as_retriever( #vector retrieval 
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": 40, "lambda_mult": 0.5}
    )

    bm25_retriever = build_bm25_retriever(vectorstore, k=k)

    return EnsembleRetriever(
        retrievers=[vector_retriever, bm25_retriever],
        weights=[0.5, 0.5]
    )