"""
rag package
===========
Standard vector-based Retrieval-Augmented Generation (RAG) module.
"""

from rag.embed import VectorStoreIndexer, build_vector_store
from rag.generator import generate_answer
from rag.pipeline import run_rag
from rag.retriever import RAGRetriever, retrieve

__all__ = [
    "run_rag",
    "RAGRetriever",
    "VectorStoreIndexer",
    "generate_answer",
    "retrieve",
    "build_vector_store",
]
