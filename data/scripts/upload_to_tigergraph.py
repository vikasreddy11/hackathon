"""
data/scripts/upload_to_tigergraph.py
======================================
Uploads the processed corpus (chunks.jsonl) to TigerGraph as a knowledge graph.

Steps:
  1. Load chunks from data/processed/chunks.jsonl
  2. Create Document + Chunk vertices
  3. Extract entities using spaCy (or regex fallback)
  4. Create Entity vertices + MENTIONS + COOCCURS_WITH edges
  5. Create PART_OF edges (Chunk -> Document)

Usage:
    python data/scripts/upload_to_tigergraph.py [--chunks PATH] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent / ".env")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_graph.tigergraph_store import TigerGraphStore
from knowledge_graph.schema import NodeData, NodeType, EdgeData, EdgeType


# ---------------------------------------------------------------------------
# Entity extraction (spaCy with regex fallback)
# ---------------------------------------------------------------------------

def get_nlp():
    try:
        import spacy
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("[upload] spaCy model not found, using regex fallback")
            return None
    except ImportError:
        return None


def extract_entities_spacy(text: str, nlp) -> list[tuple[str, str]]:
    """Return list of (normalised_text, label) pairs."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "EVENT", "FAC", "LOC", "NORP", "PRODUCT"):
            norm = ent.text.strip().lower()
            if len(norm) >= 2:
                entities.append((norm, ent.label_))
    return entities


def extract_entities_regex(text: str) -> list[tuple[str, str]]:
    """Simple capitalised-phrase extraction as fallback."""
    import re
    # Match sequences of capitalised words (2-4 words)
    pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b'
    matches = re.findall(pattern, text)
    seen = set()
    entities = []
    for m in matches:
        norm = m.strip().lower()
        if len(norm) >= 3 and norm not in seen:
            seen.add(norm)
            entities.append((norm, "MISC"))
    return entities[:20]  # cap per chunk


# ---------------------------------------------------------------------------
# Main upload pipeline
# ---------------------------------------------------------------------------

def upload(chunks_path: Path, limit: int | None = None, skip_schema: bool = False):
    print(f"[upload] Loading chunks from {chunks_path}")
    chunks = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if limit:
        chunks = chunks[:limit]

    print(f"[upload] Loaded {len(chunks)} chunks")

    # Connect to TigerGraph
    store = TigerGraphStore(auto_connect=True)

    if not skip_schema:
        print("[upload] Setting up schema (this is safe to re-run)...")
        try:
            store.setup_schema()
        except Exception as e:
            print(f"[upload] Schema setup note: {e}")

    # Load NLP
    nlp = get_nlp()
    extract_fn = (lambda t: extract_entities_spacy(t, nlp)) if nlp else extract_entities_regex

    # Track seen documents (avoid duplicate Document vertices)
    seen_docs: set[str] = set()
    total_chunks = 0
    total_entities = 0
    total_edges = 0

    print(f"[upload] Starting upload...")

    for i, chunk in enumerate(chunks):
        doc_id = chunk.get("doc_id", "")
        chunk_id = chunk.get("chunk_id", "")
        title = chunk.get("title", "")
        text = chunk.get("text", "")
        url = chunk.get("url", "") or chunk.get("source", "")

        # --- Document vertex (once per doc_id) ---
        if doc_id and doc_id not in seen_docs:
            store.add_node(NodeData(
                node_id=doc_id,
                node_type=NodeType.DOCUMENT,
                label=title,
                metadata={
                    "url": url,
                    "wikidata_qid": chunk.get("wikidata_qid", ""),
                    "wikipedia_pageid": chunk.get("wikipedia_pageid", ""),
                }
            ))
            seen_docs.add(doc_id)

        # --- Chunk vertex ---
        if chunk_id:
            store.add_node(NodeData(
                node_id=chunk_id,
                node_type=NodeType.CHUNK,
                label=title,
                source_doc_id=doc_id,
                metadata={
                    "text": text,
                    "token_count": chunk.get("token_count", 0),
                    "char_start": chunk.get("char_start", 0),
                    "char_end": chunk.get("char_end", 0),
                    "source": url,
                }
            ))
            total_chunks += 1

            # --- PART_OF edge: Chunk -> Document ---
            if doc_id:
                store.add_edge(EdgeData(
                    source_id=chunk_id,
                    target_id=doc_id,
                    edge_type=EdgeType.PART_OF,
                ))
                total_edges += 1

            # --- Extract entities ---
            entities = extract_fn(text)
            entity_ids_in_chunk = []

            for norm_text, ent_label in entities:
                eid = norm_text  # normalised text as ID
                store.add_node(NodeData(
                    node_id=eid,
                    node_type=NodeType.ENTITY,
                    label=norm_text,
                    entity_label=ent_label,
                    source_doc_id=doc_id,
                ))
                total_entities += 1

                # MENTIONS edge: Chunk -> Entity
                store.add_edge(EdgeData(
                    source_id=chunk_id,
                    target_id=eid,
                    edge_type=EdgeType.MENTIONS,
                    metadata={"chunk_id": chunk_id},
                ))
                total_edges += 1
                entity_ids_in_chunk.append(eid)

            # --- COOCCURS_WITH edges (all pairs within chunk) ---
            for j in range(len(entity_ids_in_chunk)):
                for k in range(j + 1, min(j + 6, len(entity_ids_in_chunk))):
                    store.add_edge(EdgeData(
                        source_id=entity_ids_in_chunk[j],
                        target_id=entity_ids_in_chunk[k],
                        edge_type=EdgeType.COOCCURS_WITH,
                        metadata={"chunk_id": chunk_id},
                    ))
                    total_edges += 1

        if (i + 1) % 500 == 0:
            print(f"[upload] Processed {i+1}/{len(chunks)} chunks | "
                  f"docs={len(seen_docs)} entities={total_entities} edges={total_edges}")

    # Final flush
    print("[upload] Final flush to TigerGraph...")
    store.flush()

    print(f"\n[upload] DONE!")
    print(f"  Documents : {len(seen_docs)}")
    print(f"  Chunks    : {total_chunks}")
    print(f"  Entities  : {total_entities}")
    print(f"  Edges     : {total_edges}")

    # Report graph stats
    try:
        nc = store.node_count()
        ec = store.edge_count()
        print(f"\n[TigerGraph] Vertex count: {nc}")
        print(f"[TigerGraph] Edge count  : {ec}")
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload corpus to TigerGraph")
    parser.add_argument("--chunks", type=Path, default=Path("data/processed/chunks.jsonl"))
    parser.add_argument("--limit", type=int, default=None, help="Limit number of chunks (for testing)")
    parser.add_argument("--skip-schema", action="store_true", help="Skip schema setup (if already done)")
    args = parser.parse_args()
    upload(args.chunks, limit=args.limit, skip_schema=args.skip_schema)
