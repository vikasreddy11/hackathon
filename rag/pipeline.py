"""
rag/pipeline.py
===============
Ties embedding, retrieval, and generation into one end-to-end RAG entrypoint:
    run_rag(question) -> {answer, sources, scores}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rag.embed import DEFAULT_INDEX_DIR
from rag.generator import generate_answer
from rag.retriever import RAGRetriever

_default_retriever: RAGRetriever | None = None


def get_default_retriever(index_dir: str | Path = DEFAULT_INDEX_DIR) -> RAGRetriever:
    """Singleton helper for default retriever."""
    global _default_retriever
    if _default_retriever is None or str(_default_retriever.index_dir) != str(index_dir):
        _default_retriever = RAGRetriever(index_dir=index_dir)
        _default_retriever.load_or_initialize()
    return _default_retriever


def run_rag(
    question: str,
    top_k: int = 3,
    backend: str | None = None,
    model: str | None = None,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    retriever: RAGRetriever | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run standard RAG pipeline:
      1. Embed query and retrieve top-k chunks from FAISS vector index.
      2. Construct augmented prompt with chunk contexts.
      3. Call LLM to synthesize factual answer.

    Parameters
    ----------
    question : str
        The query to answer.
    top_k : int
        Number of most similar chunks to retrieve (default: 3).
    backend : str | None
        LLM backend ('auto', 'mock', 'groq', 'openai', 'gemini', 'anthropic').
    model : str | None
        LLM model override.
    index_dir : str | Path
        Directory where FAISS vector index is stored.
    retriever : RAGRetriever | None
        Optional pre-loaded RAGRetriever instance.

    Returns
    -------
    dict[str, Any]
        {
            "answer": str,
            "sources": list[dict],
            "scores": list[float],
        }
    """
    active_retriever = retriever or get_default_retriever(index_dir=index_dir)
    retrieved_chunks = active_retriever.retrieve(query=question, top_k=top_k)

    answer = generate_answer(
        question=question,
        context_chunks=retrieved_chunks,
        backend=backend,
        model=model,
        **kwargs,
    )

    scores = [float(c.get("score", 0.0)) for c in retrieved_chunks]
    sources = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "doc_id": c.get("doc_id", ""),
            "text": c.get("text", ""),
            "source": c.get("source", ""),
            "score": float(c.get("score", 0.0)),
        }
        for c in retrieved_chunks
    ]

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
    }


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Run standard RAG on a query.")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--backend", type=str, default=None, help="LLM backend override")
    args = parser.parse_args()

    result = run_rag(question=args.question, top_k=args.top_k, backend=args.backend)
    print("\n--- RAG Output ---")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Retrieved {len(result['sources'])} sources with scores: {result['scores']}")
