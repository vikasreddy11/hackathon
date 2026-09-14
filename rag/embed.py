"""
rag/embed.py
============
Vector embedding generation and FAISS index persistence for document chunks.
Uses sentence-transformers ('all-MiniLM-L6-v2') and FAISS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DEFAULT_INDEX_DIR = Path("rag/vector_store")
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(chunks_path: str | Path = DEFAULT_CHUNKS_PATH) -> list[dict[str, Any]]:
    """Load chunks from chunks.jsonl."""
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Chunk file not found at: {path.resolve()}.\n"
            f"Please generate chunks using 'python knowledge_graph/build.py' first."
        )

    chunks: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                chunks.append(data)
            except json.JSONDecodeError as exc:
                print(f"[Warning] Skipping invalid JSON at line {line_num}: {exc}")

    if not chunks:
        raise ValueError(f"No valid chunks found in {path}")

    return chunks


class VectorStoreIndexer:
    """Handles embedding generation and FAISS index persistence."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
    ):
        self.model_name = model_name
        self.index_dir = Path(index_dir)
        self.model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build_index(
        self,
        chunks: list[dict[str, Any]],
        batch_size: int = 32,
    ) -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
        """Generate normalized embeddings and build an inner-product (cosine) FAISS index."""
        model = self._get_model()
        texts = [c.get("text", "") for c in chunks]

        # Generate embeddings as float32 numpy array
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        return index, chunks

    def save(
        self,
        index: faiss.IndexFlatIP,
        chunks: list[dict[str, Any]],
    ) -> None:
        """Save FAISS index and metadata to index_dir."""
        self.index_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.index_dir / "index.faiss"
        meta_path = self.index_dir / "metadata.json"

        faiss.write_index(index, str(index_path))

        # Store metadata mapping vector IDs to chunk data
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "count": len(chunks),
                    "chunks": chunks,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"[VectorStoreIndexer] Successfully indexed {len(chunks)} chunks.")
        print(f"  FAISS index -> {index_path}")
        print(f"  Metadata    -> {meta_path}")

    def build_and_save(
        self,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    ) -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
        """Convenience method to load chunks, build index, and save."""
        chunks = load_chunks(chunks_path)
        index, metadata = self.build_index(chunks)
        self.save(index, metadata)
        return index, metadata


def build_vector_store(
    chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[faiss.IndexFlatIP, list[dict[str, Any]]]:
    """Top-level helper to build and persist vector store."""
    indexer = VectorStoreIndexer(model_name=model_name, index_dir=index_dir)
    return indexer.build_and_save(chunks_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed chunks and build FAISS index.")
    parser.add_argument(
        "--chunks",
        type=str,
        default=str(DEFAULT_CHUNKS_PATH),
        help=f"Path to chunks.jsonl (default: {DEFAULT_CHUNKS_PATH})",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_INDEX_DIR),
        help=f"Output directory for vector store (default: {DEFAULT_INDEX_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help=f"SentenceTransformer model name (default: {DEFAULT_MODEL_NAME})",
    )
    args = parser.parse_args()
    build_vector_store(chunks_path=args.chunks, index_dir=args.out, model_name=args.model)
