"""
rag/retriever.py
================
Given a user query, embeds it and retrieves top-k most similar document chunks
along with their cosine similarity scores from the FAISS vector index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.embed import DEFAULT_CHUNKS_PATH, DEFAULT_INDEX_DIR, DEFAULT_MODEL_NAME, VectorStoreIndexer


class RAGRetriever:
    """Retrieves top-k document chunks for a query using FAISS."""

    def __init__(
        self,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        model_name: str = DEFAULT_MODEL_NAME,
        auto_build_if_missing: bool = True,
    ):
        self.index_dir = Path(index_dir)
        self.model_name = model_name
        self.auto_build_if_missing = auto_build_if_missing

        self.index: faiss.IndexFlatIP | None = None
        self.chunks: list[dict[str, Any]] = []
        self.model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def load_or_initialize(self) -> None:
        """Load persisted FAISS index and chunk metadata, or auto-build if allowed."""
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "metadata.json"

        if index_path.exists() and meta_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self.chunks = meta.get("chunks", [])
                self.model_name = meta.get("model_name", self.model_name)
            return

        if self.auto_build_if_missing and DEFAULT_CHUNKS_PATH.exists():
            print(f"[RAGRetriever] Index not found at {self.index_dir}. Auto-building from {DEFAULT_CHUNKS_PATH}...")
            indexer = VectorStoreIndexer(model_name=self.model_name, index_dir=self.index_dir)
            self.index, self.chunks = indexer.build_and_save(DEFAULT_CHUNKS_PATH)
            return

        raise FileNotFoundError(
            f"FAISS index or metadata not found at {self.index_dir}.\n"
            f"Run 'python -m rag.embed' or generate chunks to initialize the vector store."
        )

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """
        Embed the query and return top-k most similar chunks with scores.

        Parameters
        ----------
        query : str
            User query or question.
        top_k : int
            Number of top results to return.

        Returns
        -------
        list[dict[str, Any]]
            List of chunk records with similarity 'score'.
        """
        if self.index is None:
            self.load_or_initialize()

        if not self.chunks or self.index is None or self.index.ntotal == 0:
            return []

        effective_k = min(top_k, len(self.chunks))
        if effective_k <= 0:
            return []

        model = self._get_model()
        query_vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        scores, indices = self.index.search(query_vector, effective_k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk_data = self.chunks[idx]
            results.append({
                "chunk_id": chunk_data.get("chunk_id", ""),
                "doc_id": chunk_data.get("doc_id", ""),
                "text": chunk_data.get("text", ""),
                "source": chunk_data.get("source", ""),
                "score": float(score),
                "token_count": chunk_data.get("token_count", 0),
                "metadata": chunk_data.get("metadata", {}),
            })

        return results


def retrieve(
    query: str,
    top_k: int = 3,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
) -> list[dict[str, Any]]:
    """Functional convenience wrapper for chunk retrieval."""
    retriever = RAGRetriever(index_dir=index_dir, model_name=model_name)
    return retriever.retrieve(query=query, top_k=top_k)
