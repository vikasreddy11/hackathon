"""
knowledge_graph/builder.py
==========================
Reads data/processed/chunks.jsonl, runs entity/relation extraction on each
chunk, and constructs a knowledge graph using the GraphStore abstraction.

Graph topology built
--------------------
  Document node
      ↑ PART_OF ← Chunk nodes
      Chunk node → MENTIONS → Entity nodes
      Entity ↔ Entity  COOCCURS_WITH  (within same chunk)
      Entity ↔ Entity  RELATED_TO     (cross-chunk, same document)

Usage
-----
    from knowledge_graph.builder import GraphBuilder
    from knowledge_graph.extractor import SpacyExtractor
    from knowledge_graph.graph_store import NetworkXGraphStore

    store = NetworkXGraphStore()
    builder = GraphBuilder(extractor=SpacyExtractor(), store=store)
    builder.build_from_chunks("data/processed/chunks.jsonl")
    store.save("data/processed/graph.graphml", "knowledge_graph/graph.pkl")
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from knowledge_graph.extractor import BaseExtractor, ExtractedEntity
from knowledge_graph.schema import EdgeData, EdgeType, NodeData, NodeType

if TYPE_CHECKING:
    from knowledge_graph.graph_store import GraphStore


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Orchestrates the full ingestion → extraction → graph construction pipeline.

    Parameters
    ----------
    extractor : BaseExtractor
        Any extractor implementing BaseExtractor.extract().
    store : GraphStore
        Any store implementing the GraphStore interface.
    verbose : bool
        Print progress messages.
    """

    def __init__(
        self,
        extractor: BaseExtractor,
        store: "GraphStore",
        verbose: bool = True,
    ):
        self.extractor = extractor
        self.store = store
        self.verbose = verbose

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_from_chunks(self, chunks_path: str | Path) -> None:
        """
        Read chunks.jsonl and build the knowledge graph in ``self.store``.

        Parameters
        ----------
        chunks_path : str | Path
            Path to data/processed/chunks.jsonl
        """
        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            raise FileNotFoundError(f"chunks.jsonl not found: {chunks_path}")

        chunks = self._load_chunks(chunks_path)
        if not chunks:
            raise ValueError(f"No chunks found in {chunks_path}")

        self._log(f"Building graph from {len(chunks)} chunks ...")

        # Track entities per document for RELATED_TO edges
        doc_entities: dict[str, set[str]] = defaultdict(set)

        for chunk in chunks:
            doc_id = chunk["doc_id"]
            chunk_id = chunk["chunk_id"]
            text = chunk["text"]

            # 1. Ensure Document node exists
            self._ensure_document_node(doc_id, chunk.get("source", ""))

            # 2. Create Chunk node
            self._add_chunk_node(chunk)

            # 3. Link Chunk → Document (PART_OF)
            self.store.add_edge(EdgeData(
                source_id=chunk_id,
                target_id=doc_id,
                edge_type=EdgeType.PART_OF,
                metadata={"doc_id": doc_id},
            ))

            # 4. Extract entities and relations from chunk text
            try:
                entities, relations = self.extractor.extract(text)
            except Exception as exc:
                self._log(f"  WARNING: extraction failed for {chunk_id}: {exc}")
                continue

            # 5. Add Entity nodes + MENTIONS edges
            for ent in entities:
                self._ensure_entity_node(ent, doc_id)
                self.store.add_edge(EdgeData(
                    source_id=chunk_id,
                    target_id=ent.normalized,
                    edge_type=EdgeType.MENTIONS,
                    metadata={"chunk_id": chunk_id, "entity_label": ent.label},
                ))
                doc_entities[doc_id].add(ent.normalized)

            # 6. Add COOCCURS_WITH edges between entities in this chunk
            for rel in relations:
                self.store.add_edge(EdgeData(
                    source_id=rel.source_text,
                    target_id=rel.target_text,
                    edge_type=rel.edge_type,
                    weight=rel.weight,
                    metadata={"chunk_id": chunk_id, **rel.metadata},
                ))

            self._log(
                f"  {chunk_id}: {len(entities)} entities, "
                f"{len(relations)} co-occurrences"
            )

        # 7. Add RELATED_TO edges for entity pairs that share a document
        #    but were NOT already connected via COOCCURS_WITH
        self._add_related_to_edges(doc_entities)

        total_nodes = len(list(self.store._graph.nodes()))  # type: ignore[attr-defined]
        total_edges = len(list(self.store._graph.edges()))  # type: ignore[attr-defined]
        self._log(
            f"\nGraph built: {total_nodes} nodes, {total_edges} edges."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_chunks(self, path: Path) -> list[dict]:
        chunks = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks

    def _ensure_document_node(self, doc_id: str, source: str) -> None:
        node = NodeData(
            node_id=doc_id,
            node_type=NodeType.DOCUMENT,
            label=doc_id,
            source_doc_id=doc_id,
            metadata={"source": source},
        )
        self.store.add_node(node)

    def _add_chunk_node(self, chunk: dict) -> None:
        node = NodeData(
            node_id=chunk["chunk_id"],
            node_type=NodeType.CHUNK,
            label=chunk["chunk_id"],
            source_doc_id=chunk["doc_id"],
            metadata={
                "char_start": chunk.get("char_start", 0),
                "char_end": chunk.get("char_end", 0),
                "token_count": chunk.get("token_count", 0),
                "text_preview": chunk["text"][:120],
            },
        )
        self.store.add_node(node)

    def _ensure_entity_node(self, ent: ExtractedEntity, doc_id: str) -> None:
        node = NodeData(
            node_id=ent.normalized,
            node_type=NodeType.ENTITY,
            label=ent.text,
            entity_label=ent.label,
            source_doc_id=doc_id,
            metadata={},
        )
        self.store.add_node(node)

    def _add_related_to_edges(
        self, doc_entities: dict[str, set[str]]
    ) -> None:
        """
        For each document, add RELATED_TO between entity pairs that co-appear
        in the document (across chunks). Skip pairs already linked by
        COOCCURS_WITH.
        """
        related_count = 0
        for doc_id, entity_ids in doc_entities.items():
            entity_list = sorted(entity_ids)
            for i, a in enumerate(entity_list):
                for b in entity_list[i + 1:]:
                    # Only add RELATED_TO if not already directly connected
                    neighbors_a = {
                        n for n in self.store.query_neighbors(a)
                    }
                    if b not in neighbors_a:
                        self.store.add_edge(EdgeData(
                            source_id=a,
                            target_id=b,
                            edge_type=EdgeType.RELATED_TO,
                            metadata={"doc_id": doc_id},
                        ))
                        related_count += 1

        self._log(f"  Added {related_count} RELATED_TO cross-chunk edges.")

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, file=sys.stdout)
