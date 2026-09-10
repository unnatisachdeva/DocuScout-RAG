from flashrank import Ranker
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

def get_reranked_retriever(base_retriever, top_n=5):

    compressor= FlashrankRerank(
        client=Ranker(),
        top_n=top_n
    )
    return ContextualCompressionRetriever(
        base_compressor= compressor,
        base_retriever= base_retriever
    )