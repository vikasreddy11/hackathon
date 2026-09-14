"""
graphrag/pipeline.py
====================
Ties entity extraction, graph retrieval, context building, and generation
into one end-to-end GraphRAG entrypoint:
    run_graphrag(question) -> {answer, sources, subgraph}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graphrag.context_builder import build_graph_context
from graphrag.entity_extractor import EntityExtractor, extract_entities
from graphrag.generator import generate_graphrag_answer
from graphrag.graph_retriever import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_PKL_PATH,
    GraphRetriever,
)

_default_retriever: GraphRetriever | None = None
_default_extractor: EntityExtractor | None = None


def get_default_graph_retriever(
    pkl_path: str | Path = DEFAULT_PKL_PATH,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
) -> GraphRetriever:
    """Singleton getter for GraphRetriever."""
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = GraphRetriever(pkl_path=pkl_path, chunks_path=chunks_path)
    return _default_retriever


def get_default_extractor() -> EntityExtractor:
    """Singleton getter for EntityExtractor."""
    global _default_extractor
    if _default_extractor is None:
        _default_extractor = EntityExtractor()
    return _default_extractor


def run_graphrag(
    question: str,
    top_k_entities: int = 10,
    max_chunks: int = 5,
    backend: str | None = None,
    model: str | None = None,
    pkl_path: str | Path = DEFAULT_PKL_PATH,
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    retriever: GraphRetriever | None = None,
    extractor: EntityExtractor | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Run end-to-end GraphRAG pipeline:
      1. Extract entities/concepts from the question.
      2. Query knowledge graph for matched nodes, 1-hop relationships, and linked chunks.
      3. Format subgraph into structured textual context.
      4. Generate factual answer using LLM.

    Parameters
    ----------
    question : str
        The query to answer.
    top_k_entities : int
        Maximum number of entities to include in the subgraph.
    max_chunks : int
        Maximum number of linked source chunks to include.
    backend : str | None
        LLM backend override ('auto', 'mock', 'groq', 'openai', 'gemini', 'anthropic').
    model : str | None
        LLM model identifier override.
    pkl_path : str | Path
        Path to knowledge graph pickle file.
    chunks_path : str | Path
        Path to chunks.jsonl file.
    retriever : GraphRetriever | None
        Optional pre-configured GraphRetriever.
    extractor : EntityExtractor | None
        Optional pre-configured EntityExtractor.

    Returns
    -------
    dict[str, Any]
        {
            "answer": str,
            "sources": list[dict],
            "subgraph": {
                "query_entities": list[str],
                "entities": list[dict],
                "relationships": list[dict],
                "chunk_ids": list[str],
            }
        }
    """
    active_extractor = extractor or get_default_extractor()
    active_retriever = retriever or get_default_graph_retriever(pkl_path=pkl_path, chunks_path=chunks_path)

    # 1. Extract candidate entities from question
    query_entities = active_extractor.extract_entities(question)

    # 2. Retrieve subgraph from knowledge graph
    subgraph_data = active_retriever.retrieve_subgraph(
        entity_queries=query_entities,
        max_entities=top_k_entities,
        max_chunks=max_chunks,
    )

    # 3. Format subgraph context
    context_text = build_graph_context(subgraph_data, max_chunks=max_chunks)

    # 4. Generate answer
    answer = generate_graphrag_answer(
        question=question,
        graph_context=context_text,
        backend=backend,
        model=model,
        **kwargs,
    )

    # Clean sources structure
    sources = [
        {
            "chunk_id": c.get("chunk_id", ""),
            "doc_id": c.get("doc_id", ""),
            "text": c.get("text", ""),
            "source": c.get("source", ""),
        }
        for c in subgraph_data.get("chunks", [])
    ]

    return {
        "answer": answer,
        "sources": sources,
        "subgraph": {
            "query_entities": subgraph_data.get("query_entities", []),
            "entities": subgraph_data.get("entities", []),
            "relationships": subgraph_data.get("relationships", []),
            "chunk_ids": subgraph_data.get("chunk_ids", []),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GraphRAG on a query.")
    parser.add_argument("question", type=str, help="Question to ask")
    parser.add_argument("--top-k-entities", type=int, default=10, help="Max entities to expand")
    parser.add_argument("--max-chunks", type=int, default=5, help="Max source chunks to include")
    parser.add_argument("--backend", type=str, default=None, help="LLM backend override")
    args = parser.parse_args()

    result = run_graphrag(
        question=args.question,
        top_k_entities=args.top_k_entities,
        max_chunks=args.max_chunks,
        backend=args.backend,
    )
    print("\n--- GraphRAG Output ---")
    print(f"Answer:\n{result['answer']}\n")
    print(f"Entities in subgraph: {len(result['subgraph']['entities'])}")
    print(f"Relationships in subgraph: {len(result['subgraph']['relationships'])}")
    print(f"Sources linked: {len(result['sources'])}")
