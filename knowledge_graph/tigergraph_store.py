"""
knowledge_graph/tigergraph_store.py
=====================================
TigerGraph backend implementation of the GraphStore interface.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from knowledge_graph.schema import EdgeData, EdgeType, NodeData, NodeType
from knowledge_graph.graph_store import GraphStore

try:
    import pyTigerGraph as tg
except ImportError:
    tg = None


class TigerGraphStore(GraphStore):
    """
    Knowledge graph backed by TigerGraph via pyTigerGraph.
    Credentials from env: TG_HOST, TG_GRAPHNAME, TG_SECRET, TIGERGRAPH_USERNAME, TIGERGRAPH_PASSWORD
    """

    BATCH_SIZE = 500

    def __init__(self, auto_connect: bool = True):
        if tg is None:
            raise ImportError("pyTigerGraph not installed. Run: pip install pyTigerGraph")

        self._host = os.environ.get("TG_HOST", "")
        self._graphname = os.environ.get("TG_GRAPHNAME", "AgenticGraphRAG")
        self._secret = os.environ.get("TG_SECRET", "")
        self._username = os.environ.get("TIGERGRAPH_USERNAME", "test1")
        self._password = os.environ.get("TIGERGRAPH_PASSWORD", "")
        self._conn = None

        self._pending_vertices: dict[str, list[dict]] = {
            "Document": [], "Chunk": [], "Entity": []
        }
        self._pending_edges: list[tuple] = []

        if auto_connect:
            self.connect()

    def connect(self) -> None:
        print(f"[TigerGraphStore] Connecting to {self._host} / graph={self._graphname}")
        try:
            self._conn = tg.TigerGraphConnection(
                host=self._host,
                graphname=self._graphname,
                username=self._username,
                password=self._password,
                gsqlSecret=self._secret,
            )
            print("[TigerGraphStore] Connected successfully.")
        except Exception as exc:
            print(f"[TigerGraphStore] Connection failed: {exc}", file=sys.stderr)
            raise

    def setup_schema(self) -> None:
        """Create graph schema on TigerGraph (run once)."""
        print("[TigerGraphStore] Setting up schema...")
        gsql_schema = """
CREATE VERTEX Document (PRIMARY_ID doc_id STRING, title STRING DEFAULT "", url STRING DEFAULT "", wikidata_qid STRING DEFAULT "", wikipedia_pageid STRING DEFAULT "") WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE VERTEX Chunk (PRIMARY_ID chunk_id STRING, doc_id STRING DEFAULT "", title STRING DEFAULT "", text STRING DEFAULT "", token_count INT DEFAULT 0, char_start INT DEFAULT 0, char_end INT DEFAULT 0, source STRING DEFAULT "") WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE VERTEX Entity (PRIMARY_ID entity_id STRING, label STRING DEFAULT "", entity_label STRING DEFAULT "", mention_count INT DEFAULT 1) WITH STATS="OUTDEGREE_BY_EDGETYPE", PRIMARY_ID_AS_ATTRIBUTE="true"
CREATE DIRECTED EDGE PART_OF (FROM Chunk, TO Document, weight FLOAT DEFAULT 1.0) WITH REVERSE_EDGE="PART_OF_REVERSE"
CREATE DIRECTED EDGE MENTIONS (FROM Chunk, TO Entity, weight FLOAT DEFAULT 1.0) WITH REVERSE_EDGE="MENTIONS_REVERSE"
CREATE UNDIRECTED EDGE COOCCURS_WITH (FROM Entity, TO Entity, weight FLOAT DEFAULT 1.0, chunk_id STRING DEFAULT "")
CREATE UNDIRECTED EDGE RELATED_TO (FROM Entity, TO Entity, weight FLOAT DEFAULT 1.0, doc_id STRING DEFAULT "")
"""
        try:
            result = self._conn.gsql(
                f"USE GLOBAL\n{gsql_schema}\nCREATE GRAPH {self._graphname} "
                f"(Document, Chunk, Entity, PART_OF, PART_OF_REVERSE, MENTIONS, MENTIONS_REVERSE, COOCCURS_WITH, RELATED_TO)"
            )
            print(f"[TigerGraphStore] Schema result: {result}")
        except Exception as exc:
            print(f"[TigerGraphStore] Schema note (may already exist): {exc}")
        print("[TigerGraphStore] Schema setup done.")

    def add_node(self, node: NodeData) -> None:
        if node.node_type == NodeType.DOCUMENT:
            self._pending_vertices["Document"].append({
                "doc_id": node.node_id,
                "title": node.label,
                "url": node.metadata.get("url", ""),
                "wikidata_qid": node.metadata.get("wikidata_qid", ""),
                "wikipedia_pageid": str(node.metadata.get("wikipedia_pageid", "")),
            })
        elif node.node_type == NodeType.CHUNK:
            self._pending_vertices["Chunk"].append({
                "chunk_id": node.node_id,
                "doc_id": node.source_doc_id or "",
                "title": node.label,
                "text": node.metadata.get("text", ""),
                "token_count": int(node.metadata.get("token_count", 0)),
                "char_start": int(node.metadata.get("char_start", 0)),
                "char_end": int(node.metadata.get("char_end", 0)),
                "source": node.metadata.get("source", ""),
            })
        elif node.node_type == NodeType.ENTITY:
            self._pending_vertices["Entity"].append({
                "entity_id": node.node_id,
                "label": node.label,
                "entity_label": node.entity_label or "MISC",
                "mention_count": 1,
            })
        self._auto_flush()

    def add_edge(self, edge: EdgeData) -> None:
        edge_map = {
            EdgeType.PART_OF: ("Chunk", "PART_OF", "Document"),
            EdgeType.MENTIONS: ("Chunk", "MENTIONS", "Entity"),
            EdgeType.COOCCURS_WITH: ("Entity", "COOCCURS_WITH", "Entity"),
            EdgeType.RELATED_TO: ("Entity", "RELATED_TO", "Entity"),
        }
        if edge.edge_type not in edge_map:
            return
        from_type, etype, to_type = edge_map[edge.edge_type]

        # Only send attributes that exist in the TG schema per edge type
        if etype in ("PART_OF", "PART_OF_REVERSE", "MENTIONS", "MENTIONS_REVERSE"):
            attrs = {"weight": edge.weight}
        elif etype == "COOCCURS_WITH":
            attrs = {"weight": edge.weight, "chunk_id": edge.metadata.get("chunk_id", "")}
        elif etype == "RELATED_TO":
            attrs = {"weight": edge.weight, "doc_id": edge.metadata.get("doc_id", "")}
        else:
            attrs = {"weight": edge.weight}

        self._pending_edges.append((from_type, edge.source_id, etype, to_type, edge.target_id, attrs))
        self._auto_flush()


    def _auto_flush(self) -> None:
        total = sum(len(v) for v in self._pending_vertices.values()) + len(self._pending_edges)
        if total >= self.BATCH_SIZE:
            self.flush()

    def flush(self) -> None:
        # Primary key field name per vertex type
        pk_map = {"Document": "doc_id", "Chunk": "chunk_id", "Entity": "entity_id"}

        for vtype, records in self._pending_vertices.items():
            if not records:
                continue
            pk = pk_map.get(vtype, "id")
            # upsertVertices expects list of (primary_id, attrs_dict) tuples
            tuples = [(r[pk], {k: v for k, v in r.items() if k != pk}) for r in records]
            try:
                result = self._conn.upsertVertices(vtype, tuples)
                print(f"[TigerGraphStore] Upserted {len(tuples)} {vtype} -> {result}")
            except Exception as exc:
                print(f"[TigerGraphStore] Error upserting {vtype}: {exc}", file=sys.stderr)
            self._pending_vertices[vtype] = []

        if self._pending_edges:
            edge_groups: dict[str, list] = {}
            for from_type, from_id, etype, to_type, to_id, attrs in self._pending_edges:
                key = f"{from_type}|{etype}|{to_type}"
                if key not in edge_groups:
                    edge_groups[key] = []
                edge_groups[key].append((from_id, to_id, attrs))
            for key, edges in edge_groups.items():
                from_type, etype, to_type = key.split("|")
                try:
                    result = self._conn.upsertEdges(from_type, etype, to_type, edges)
                    print(f"[TigerGraphStore] Upserted {len(edges)} {etype} edges -> {result}")
                except Exception as exc:
                    print(f"[TigerGraphStore] Error upserting {etype}: {exc}", file=sys.stderr)
            self._pending_edges = []


    def query_neighbors(self, node_id: str, edge_type: EdgeType | None = None) -> list[str]:
        try:
            for vtype in ("Document", "Chunk", "Entity"):
                try:
                    neighbors = self._conn.getVertexNeighbors(vtype, node_id)
                    if neighbors:
                        return [n.get("v_id", "") for n in neighbors]
                except Exception:
                    continue
        except Exception as exc:
            print(f"[TigerGraphStore] query_neighbors error: {exc}", file=sys.stderr)
        return []

    def query_by_entity(self, entity_text: str) -> list[NodeData]:
        try:
            results = self._conn.getVerticesByAttribute("Entity", "label", entity_text)
            return [
                NodeData(
                    node_id=r.get("v_id", ""),
                    node_type=NodeType.ENTITY,
                    label=r.get("attributes", {}).get("label", ""),
                    entity_label=r.get("attributes", {}).get("entity_label"),
                )
                for r in results
            ]
        except Exception as exc:
            print(f"[TigerGraphStore] query_by_entity error: {exc}", file=sys.stderr)
            return []

    def get_chunks_for_entity(self, entity_id: str, limit: int = 10) -> list[dict]:
        """Return chunks that MENTION a given entity (reverse MENTIONS edge)."""
        try:
            results = self._conn.getEdges("Entity", entity_id, "MENTIONS_REVERSE")
            chunk_ids = [r.get("to_id", "") for r in results[:limit]]
            chunks = []
            for cid in chunk_ids:
                vdata = self._conn.getVerticesById("Chunk", [cid])
                if vdata:
                    attrs = vdata[0].get("attributes", {})
                    chunks.append({
                        "chunk_id": cid,
                        "doc_id": attrs.get("doc_id", ""),
                        "text": attrs.get("text", ""),
                        "title": attrs.get("title", ""),
                        "token_count": attrs.get("token_count", 0),
                    })
            return chunks
        except Exception as exc:
            print(f"[TigerGraphStore] get_chunks_for_entity error: {exc}", file=sys.stderr)
            return []

    def get_entity_neighbors(self, entity_id: str, limit: int = 20) -> list[dict]:
        """Return co-occurring entities."""
        try:
            results = self._conn.getEdges("Entity", entity_id, "COOCCURS_WITH")
            return [
                {"entity_id": r.get("to_id", ""), "weight": r.get("attributes", {}).get("weight", 1.0)}
                for r in results[:limit]
            ]
        except Exception as exc:
            print(f"[TigerGraphStore] get_entity_neighbors error: {exc}", file=sys.stderr)
            return []

    def save(self, graphml_path=None, pkl_path=None) -> None:
        self.flush()
        print("[TigerGraphStore] Data flushed to TigerGraph.")

    def load(self, graphml_path=None, pkl_path=None) -> None:
        if self._conn is None:
            self.connect()

    def node_count(self) -> int:
        try:
            counts = self._conn.getVertexCount("*")
            return sum(counts.values()) if isinstance(counts, dict) else int(counts)
        except Exception:
            return -1

    def edge_count(self) -> int:
        try:
            counts = self._conn.getEdgeCount()
            return sum(counts.values()) if isinstance(counts, dict) else int(counts)
        except Exception:
            return -1
