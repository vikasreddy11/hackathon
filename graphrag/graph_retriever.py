"""
graphrag/graph_retriever.py
===========================
Retrieves a structured subgraph (entities, relationships, and linked chunks)
from the Knowledge Graph for a set of entity mentions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from knowledge_graph.graph_store import GraphStore, NetworkXGraphStore
from knowledge_graph.schema import EdgeType, NodeData, NodeType

DEFAULT_PKL_PATH = Path("knowledge_graph/graph.pkl")
DEFAULT_CHUNKS_PATH = Path("data/processed/chunks.jsonl")


class GraphRetriever:
    """Retrieves entity neighborhoods, relations, and supporting chunks from GraphStore."""

    def __init__(
        self,
        pkl_path: str | Path = DEFAULT_PKL_PATH,
        chunks_path: str | Path = DEFAULT_CHUNKS_PATH,
        store: GraphStore | None = None,
    ):
        self.pkl_path = Path(pkl_path)
        self.chunks_path = Path(chunks_path)
        self.store = store
        self._chunks_cache: dict[str, dict[str, Any]] | None = None

    def _ensure_store(self) -> GraphStore:
        if self.store is None:
            store = NetworkXGraphStore()
            if self.pkl_path.exists():
                store.load(pkl_path=self.pkl_path)
            else:
                raise FileNotFoundError(
                    f"Knowledge graph not found at {self.pkl_path}.\n"
                    f"Please run 'python knowledge_graph/build.py' to generate the graph."
                )
            self.store = store
        return self.store

    def _ensure_chunks_cache(self) -> dict[str, dict[str, Any]]:
        if self._chunks_cache is None:
            self._chunks_cache = {}
            if self.chunks_path.exists():
                with open(self.chunks_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                cid = data.get("chunk_id")
                                if cid:
                                    self._chunks_cache[cid] = data
                            except Exception:
                                pass
        return self._chunks_cache

    def get_chunk_data(self, chunk_id: str) -> dict[str, Any]:
        """Fetch full chunk text and metadata by chunk_id."""
        cache = self._ensure_chunks_cache()
        if chunk_id in cache:
            return cache[chunk_id]

        # Fallback: check node metadata in graph
        store = self._ensure_store()
        node = store.get_node(chunk_id) if hasattr(store, "get_node") else None
        if node:
            return {
                "chunk_id": node.node_id,
                "doc_id": node.source_doc_id or "",
                "text": node.metadata.get("text_preview", node.label),
                "source": node.metadata.get("source", ""),
                "token_count": int(node.metadata.get("token_count", 0)),
            }

        return {"chunk_id": chunk_id, "doc_id": "", "text": "", "source": ""}

    def retrieve_subgraph(
        self,
        entity_queries: list[str],
        max_entities: int = 15,
        max_chunks: int = 5,
    ) -> dict[str, Any]:
        """
        Given entity query strings, retrieve matched nodes, connected entities,
        relationships, and linked source chunks.

        Parameters
        ----------
        entity_queries : list[str]
            Entity terms extracted from user query.
        max_entities : int
            Max total entity nodes to include in subgraph.
        max_chunks : int
            Max chunk texts to link to the subgraph context.

        Returns
        -------
        dict[str, Any]
            {
                "query_entities": list[str],
                "entities": list[dict],
                "relationships": list[dict],
                "chunk_ids": list[str],
                "chunks": list[dict],
            }
        """
        store = self._ensure_store()

        matched_node_ids: set[str] = set()
        matched_nodes: dict[str, NodeData] = {}

        # 1. Look up each entity in the graph
        for eq in entity_queries:
            eq_clean = eq.strip()
            if not eq_clean:
                continue

            # Try exact node_id lookup first
            direct = store.get_node(eq_clean.lower()) if hasattr(store, "get_node") else None
            if direct:
                matched_node_ids.add(direct.node_id)
                matched_nodes[direct.node_id] = direct

            # Partial match search
            results = store.query_by_entity(eq_clean)
            for n in results:
                matched_node_ids.add(n.node_id)
                matched_nodes[n.node_id] = n

        # 2. Expand 1-hop neighborhood for matched entities
        discovered_entities: dict[str, dict[str, Any]] = {}
        relationships: list[dict[str, Any]] = []
        linked_chunk_ids: list[str] = []
        seen_edges: set[tuple[str, str, str]] = set()

        for nid in list(matched_node_ids):
            node = matched_nodes[nid]
            if node.node_type == NodeType.ENTITY:
                discovered_entities[nid] = {
                    "node_id": node.node_id,
                    "label": node.label,
                    "entity_label": node.entity_label or "UNKNOWN",
                    "source_doc_id": node.source_doc_id or "",
                    "is_seed": True,
                }
            elif node.node_type == NodeType.CHUNK:
                if nid not in linked_chunk_ids:
                    linked_chunk_ids.append(nid)

            # Query neighbors
            nbr_ids = store.query_neighbors(nid)
            for nbr_id in nbr_ids:
                nbr_node = store.get_node(nbr_id) if hasattr(store, "get_node") else None
                if not nbr_node:
                    continue

                if nbr_node.node_type == NodeType.CHUNK:
                    if nbr_id not in linked_chunk_ids:
                        linked_chunk_ids.append(nbr_id)
                elif nbr_node.node_type == NodeType.ENTITY:
                    if nbr_id not in discovered_entities and len(discovered_entities) < max_entities:
                        discovered_entities[nbr_id] = {
                            "node_id": nbr_node.node_id,
                            "label": nbr_node.label,
                            "entity_label": nbr_node.entity_label or "UNKNOWN",
                            "source_doc_id": nbr_node.source_doc_id or "",
                            "is_seed": False,
                        }

                # Extract edge relationships between nid and nbr_id
                if hasattr(store, "_graph"):
                    g = store._graph
                    for u, v in [(nid, nbr_id), (nbr_id, nid)]:
                        edge_data = g.get_edge_data(u, v, default={})
                        for _, d in edge_data.items():
                            e_type = d.get("edge_type", "RELATED_TO")
                            edge_key = (u, v, e_type)
                            if edge_key not in seen_edges:
                                seen_edges.add(edge_key)
                                relationships.append({
                                    "source": u,
                                    "target": v,
                                    "edge_type": e_type,
                                    "weight": float(d.get("weight", 1.0)),
                                    "metadata": {k: v for k, v in d.items() if k not in ("edge_type", "weight")},
                                })

        # Cap linked chunks
        selected_chunk_ids = linked_chunk_ids[:max_chunks]
        resolved_chunks = [self.get_chunk_data(cid) for cid in selected_chunk_ids]

        return {
            "query_entities": entity_queries,
            "entities": list(discovered_entities.values()),
            "relationships": relationships,
            "chunk_ids": selected_chunk_ids,
            "chunks": resolved_chunks,
        }
