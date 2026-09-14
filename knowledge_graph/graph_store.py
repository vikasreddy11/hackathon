"""
knowledge_graph/graph_store.py
==============================
Persistence layer for the knowledge graph.

Architecture
------------
GraphStore              — Abstract interface (swap-in point for Neo4j, etc.)
NetworkXGraphStore      — Default concrete implementation (NetworkX + GraphML/pickle)

Interface contract (stable — do NOT change method signatures):

    store = NetworkXGraphStore()
    store.add_node(NodeData(...))
    store.add_edge(EdgeData(...))
    neighbors = store.query_neighbors("openai")          # list[str] of node_ids
    results   = store.query_by_entity("openai")          # list[NodeData]
    store.save(graphml_path, pkl_path)
    store.load(graphml_path, pkl_path)

See interfaces/data_kg_interface.md for full documentation.
"""

from __future__ import annotations

import abc
import pickle
from pathlib import Path
from typing import Any

import networkx as nx

from knowledge_graph.schema import EdgeData, EdgeType, NodeData, NodeType


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class GraphStore(abc.ABC):
    """
    Abstract graph persistence interface.

    All concrete backends (NetworkX, Neo4j, …) must implement these methods.
    Callers in rag/, graphrag/, agents/ should depend only on this interface.
    """

    @abc.abstractmethod
    def add_node(self, node: NodeData) -> None:
        """
        Add or update a node in the graph.

        If a node with the same node_id already exists, its attributes are
        updated (merge semantics — existing attributes not in the new NodeData
        are preserved).
        """
        ...

    @abc.abstractmethod
    def add_edge(self, edge: EdgeData) -> None:
        """
        Add or update a directed edge between two nodes.

        Source and target nodes must already exist (call add_node first).
        If the edge already exists, its weight is incremented (additive
        co-occurrence semantics).
        """
        ...

    @abc.abstractmethod
    def query_neighbors(self, node_id: str, edge_type: EdgeType | None = None) -> list[str]:
        """
        Return the node_ids of all direct neighbours of node_id.

        Parameters
        ----------
        node_id : str
            The node to query.
        edge_type : EdgeType | None
            If provided, filter to edges of this type only.

        Returns
        -------
        list[str]
            node_ids of adjacent nodes (both in- and out-neighbours for
            directed graphs).
        """
        ...

    @abc.abstractmethod
    def query_by_entity(self, entity_text: str) -> list[NodeData]:
        """
        Return all NodeData objects whose label or node_id contains
        entity_text (case-insensitive partial match).

        Returns
        -------
        list[NodeData]
        """
        ...

    @abc.abstractmethod
    def save(self, graphml_path: str | Path, pkl_path: str | Path) -> None:
        """Persist the graph to disk (GraphML + pickle)."""
        ...

    @abc.abstractmethod
    def load(self, graphml_path: str | Path | None = None, pkl_path: str | Path | None = None) -> None:
        """Load the graph from disk (prefer pkl_path if both given)."""
        ...

    @abc.abstractmethod
    def node_count(self) -> int:
        """Return the total number of nodes."""
        ...

    @abc.abstractmethod
    def edge_count(self) -> int:
        """Return the total number of edges."""
        ...


# ---------------------------------------------------------------------------
# NetworkX concrete implementation
# ---------------------------------------------------------------------------

class NetworkXGraphStore(GraphStore):
    """
    Knowledge graph backed by a NetworkX MultiDiGraph.

    Persistence
    -----------
    - GraphML  : human-readable, compatible with Gephi / yEd / graph tools
    - Pickle   : preserves full Python objects (NodeData, EdgeData); faster
                 for large graphs; used as the primary runtime format

    Swap-out path
    -------------
    To replace with Neo4j, implement a ``Neo4jGraphStore(GraphStore)`` class
    in this file and pass it to ``GraphBuilder`` — no other code changes needed.
    """

    def __init__(self):
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # GraphStore interface implementation
    # ------------------------------------------------------------------

    def add_node(self, node: NodeData) -> None:
        """Add or merge a node into the graph."""
        if self._graph.has_node(node.node_id):
            # Merge: update only non-None attributes
            existing = self._graph.nodes[node.node_id]
            attrs = node.to_dict()
            for k, v in attrs.items():
                if v:  # Don't overwrite with empty string
                    existing[k] = v
        else:
            self._graph.add_node(node.node_id, **node.to_dict())

    def add_edge(self, edge: EdgeData) -> None:
        """Add or increment an edge in the graph."""
        # Ensure both nodes exist (add minimal stubs if not)
        for nid in (edge.source_id, edge.target_id):
            if not self._graph.has_node(nid):
                stub = NodeData(
                    node_id=nid,
                    node_type=NodeType.ENTITY,
                    label=nid,
                )
                self._graph.add_node(nid, **stub.to_dict())

        # Check for an existing edge of the same type
        existing_edges = self._graph.get_edge_data(
            edge.source_id, edge.target_id, default={}
        )
        for key, data in existing_edges.items():
            if data.get("edge_type") == edge.edge_type.value:
                # Increment weight
                self._graph[edge.source_id][edge.target_id][key]["weight"] = (
                    float(data.get("weight", 1.0)) + edge.weight
                )
                return

        # New edge
        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            **edge.to_dict(),
        )

    def query_neighbors(
        self, node_id: str, edge_type: EdgeType | None = None
    ) -> list[str]:
        """Return adjacent node_ids (successors + predecessors)."""
        if not self._graph.has_node(node_id):
            return []

        successors = set(self._graph.successors(node_id))
        predecessors = set(self._graph.predecessors(node_id))
        neighbors = successors | predecessors
        neighbors.discard(node_id)  # No self-loops

        if edge_type is None:
            return list(neighbors)

        # Filter by edge type
        filtered: set[str] = set()
        for nbr in neighbors:
            for src, tgt in [(node_id, nbr), (nbr, node_id)]:
                edge_data = self._graph.get_edge_data(src, tgt, default={})
                for _, data in edge_data.items():
                    if data.get("edge_type") == edge_type.value:
                        filtered.add(nbr)
        return list(filtered)

    def query_by_entity(self, entity_text: str) -> list[NodeData]:
        """Case-insensitive partial-match search on node label and node_id."""
        needle = entity_text.strip().lower()
        results: list[NodeData] = []
        for node_id, attrs in self._graph.nodes(data=True):
            label = str(attrs.get("label", "")).lower()
            nid_lower = str(node_id).lower()
            if needle in label or needle in nid_lower:
                results.append(NodeData.from_dict({**attrs, "node_id": node_id}))
        return results

    def save(self, graphml_path: str | Path, pkl_path: str | Path) -> None:
        """Save graph to GraphML (text) and pickle (binary)."""
        graphml_path = Path(graphml_path)
        pkl_path = Path(pkl_path)

        graphml_path.parent.mkdir(parents=True, exist_ok=True)
        pkl_path.parent.mkdir(parents=True, exist_ok=True)

        # GraphML — serialise all node/edge attributes as strings
        nx.write_graphml(self._graph, str(graphml_path))
        print(f"[GraphStore] Saved GraphML → {graphml_path}")

        # Pickle
        with open(pkl_path, "wb") as fh:
            pickle.dump(self._graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[GraphStore] Saved pickle  → {pkl_path}")

    def load(
        self,
        graphml_path: str | Path | None = None,
        pkl_path: str | Path | None = None,
    ) -> None:
        """Load graph from pickle (preferred) or GraphML."""
        if pkl_path and Path(pkl_path).exists():
            with open(pkl_path, "rb") as fh:
                self._graph = pickle.load(fh)
            print(f"[GraphStore] Loaded pickle  ← {pkl_path}")
        elif graphml_path and Path(graphml_path).exists():
            self._graph = nx.read_graphml(str(graphml_path))
            print(f"[GraphStore] Loaded GraphML ← {graphml_path}")
        else:
            raise FileNotFoundError(
                f"No graph file found at pkl={pkl_path}, graphml={graphml_path}"
            )

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Convenience / introspection helpers
    # ------------------------------------------------------------------

    def get_node(self, node_id: str) -> NodeData | None:
        """Return a NodeData for node_id, or None if not present."""
        if not self._graph.has_node(node_id):
            return None
        attrs = self._graph.nodes[node_id]
        return NodeData.from_dict({**attrs, "node_id": node_id})

    def nodes_by_type(self, node_type: NodeType) -> list[NodeData]:
        """Return all nodes of a given type."""
        return [
            NodeData.from_dict({**attrs, "node_id": nid})
            for nid, attrs in self._graph.nodes(data=True)
            if attrs.get("node_type") == node_type.value
        ]

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging / debugging."""
        type_counts: dict[str, int] = {}
        for _, attrs in self._graph.nodes(data=True):
            nt = attrs.get("node_type", "UNKNOWN")
            type_counts[nt] = type_counts.get(nt, 0) + 1

        edge_counts: dict[str, int] = {}
        for _, _, attrs in self._graph.edges(data=True):
            et = attrs.get("edge_type", "UNKNOWN")
            edge_counts[et] = edge_counts.get(et, 0) + 1

        return {
            "total_nodes": self.node_count(),
            "total_edges": self.edge_count(),
            "node_type_counts": type_counts,
            "edge_type_counts": edge_counts,
        }
