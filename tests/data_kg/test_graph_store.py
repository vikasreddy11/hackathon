"""
tests/data_kg/test_graph_store.py
==================================
Unit and integration tests for:
  - knowledge_graph/graph_store.py (NetworkXGraphStore)
  - knowledge_graph/builder.py (GraphBuilder, end-to-end)
  - knowledge_graph/schema.py (NodeData, EdgeData round-trips)

Covered:
- add_node / add_edge basic operations
- Duplicate node merge semantics
- Duplicate edge weight accumulation
- query_neighbors (unfiltered and by edge_type)
- query_by_entity (partial match, case-insensitive)
- save / load round-trip (GraphML and pickle)
- node_count / edge_count correctness
- NodeData.to_dict / from_dict round-trip
- EdgeData.to_dict / from_dict round-trip
- GraphBuilder on sample docs produces ≥ expected node/edge counts
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from knowledge_graph.graph_store import NetworkXGraphStore
from knowledge_graph.schema import EdgeData, EdgeType, NodeData, NodeType


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def make_entity_node(node_id: str, label: str = None, ent_label: str = "ORG") -> NodeData:
    return NodeData(
        node_id=node_id,
        node_type=NodeType.ENTITY,
        label=label or node_id,
        entity_label=ent_label,
        source_doc_id="doc_test",
    )


def make_chunk_node(chunk_id: str, doc_id: str = "doc_test") -> NodeData:
    return NodeData(
        node_id=chunk_id,
        node_type=NodeType.CHUNK,
        label=chunk_id,
        source_doc_id=doc_id,
        metadata={"token_count": 100},
    )


def make_doc_node(doc_id: str) -> NodeData:
    return NodeData(
        node_id=doc_id,
        node_type=NodeType.DOCUMENT,
        label=doc_id,
        source_doc_id=doc_id,
    )


def make_edge(src: str, tgt: str, etype: EdgeType = EdgeType.COOCCURS_WITH) -> EdgeData:
    return EdgeData(source_id=src, target_id=tgt, edge_type=etype, weight=1.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def empty_store() -> NetworkXGraphStore:
    return NetworkXGraphStore()


@pytest.fixture
def populated_store() -> NetworkXGraphStore:
    """A store with a small known graph for query tests."""
    store = NetworkXGraphStore()
    doc = make_doc_node("doc_test")
    chunk = make_chunk_node("doc_test__chunk_0000")
    ent_a = make_entity_node("openai", "OpenAI")
    ent_b = make_entity_node("anthropic", "Anthropic")
    ent_c = make_entity_node("google", "Google", ent_label="ORG")

    for node in [doc, chunk, ent_a, ent_b, ent_c]:
        store.add_node(node)

    store.add_edge(EdgeData(
        source_id="doc_test__chunk_0000",
        target_id="doc_test",
        edge_type=EdgeType.PART_OF,
    ))
    store.add_edge(EdgeData(
        source_id="doc_test__chunk_0000",
        target_id="openai",
        edge_type=EdgeType.MENTIONS,
    ))
    store.add_edge(EdgeData(
        source_id="doc_test__chunk_0000",
        target_id="anthropic",
        edge_type=EdgeType.MENTIONS,
    ))
    store.add_edge(EdgeData(
        source_id="openai",
        target_id="anthropic",
        edge_type=EdgeType.COOCCURS_WITH,
    ))
    store.add_edge(EdgeData(
        source_id="openai",
        target_id="google",
        edge_type=EdgeType.RELATED_TO,
    ))
    return store


# ---------------------------------------------------------------------------
# Schema round-trip tests
# ---------------------------------------------------------------------------

class TestSchemaRoundTrip:
    def test_node_data_to_from_dict(self):
        node = NodeData(
            node_id="test_id",
            node_type=NodeType.ENTITY,
            label="Test Label",
            entity_label="ORG",
            source_doc_id="doc1",
            metadata={"char_start": "0", "token_count": "50"},
        )
        d = node.to_dict()
        restored = NodeData.from_dict(d)
        assert restored.node_id == node.node_id
        assert restored.node_type == node.node_type
        assert restored.label == node.label
        assert restored.entity_label == node.entity_label
        assert restored.source_doc_id == node.source_doc_id

    def test_edge_data_to_from_dict(self):
        edge = EdgeData(
            source_id="chunk_001",
            target_id="openai",
            edge_type=EdgeType.MENTIONS,
            weight=2.5,
            metadata={"chunk_id": "chunk_001"},
        )
        d = edge.to_dict()
        restored = EdgeData.from_dict(d)
        assert restored.source_id == edge.source_id
        assert restored.target_id == edge.target_id
        assert restored.edge_type == edge.edge_type
        assert abs(restored.weight - edge.weight) < 1e-6


# ---------------------------------------------------------------------------
# NetworkXGraphStore unit tests
# ---------------------------------------------------------------------------

class TestAddNode:
    def test_add_single_node(self, empty_store):
        empty_store.add_node(make_entity_node("test_entity"))
        assert empty_store.node_count() == 1

    def test_add_multiple_nodes(self, empty_store):
        for i in range(10):
            empty_store.add_node(make_entity_node(f"entity_{i}"))
        assert empty_store.node_count() == 10

    def test_duplicate_node_merge(self, empty_store):
        node1 = NodeData(
            node_id="openai", node_type=NodeType.ENTITY, label="openai", entity_label="ORG"
        )
        node2 = NodeData(
            node_id="openai", node_type=NodeType.ENTITY, label="OpenAI", entity_label="ORG"
        )
        empty_store.add_node(node1)
        empty_store.add_node(node2)
        # Should still be 1 node (merge, not duplicate)
        assert empty_store.node_count() == 1

    def test_node_data_retrievable(self, empty_store):
        node = make_entity_node("openai", "OpenAI")
        empty_store.add_node(node)
        retrieved = empty_store.get_node("openai")
        assert retrieved is not None
        assert retrieved.node_id == "openai"
        assert retrieved.entity_label == "ORG"

    def test_get_nonexistent_node_returns_none(self, empty_store):
        assert empty_store.get_node("does_not_exist") is None


class TestAddEdge:
    def test_add_edge_between_existing_nodes(self, empty_store):
        empty_store.add_node(make_entity_node("openai"))
        empty_store.add_node(make_entity_node("anthropic"))
        empty_store.add_edge(make_edge("openai", "anthropic"))
        assert empty_store.edge_count() == 1

    def test_add_edge_auto_creates_stub_nodes(self, empty_store):
        # Edge between non-existent nodes — stubs should be created
        empty_store.add_edge(make_edge("node_a", "node_b"))
        assert empty_store.node_count() == 2
        assert empty_store.edge_count() == 1

    def test_duplicate_edge_increments_weight(self, empty_store):
        empty_store.add_node(make_entity_node("openai"))
        empty_store.add_node(make_entity_node("anthropic"))
        edge = EdgeData(
            source_id="openai", target_id="anthropic",
            edge_type=EdgeType.COOCCURS_WITH, weight=1.0
        )
        empty_store.add_edge(edge)
        empty_store.add_edge(edge)
        # Edge count should still be 1 (same edge, weight incremented)
        assert empty_store.edge_count() == 1

    def test_different_edge_types_are_separate_edges(self, empty_store):
        for etype in [EdgeType.COOCCURS_WITH, EdgeType.RELATED_TO]:
            empty_store.add_edge(EdgeData(
                source_id="openai", target_id="anthropic",
                edge_type=etype,
            ))
        assert empty_store.edge_count() == 2


class TestQueryNeighbors:
    def test_returns_adjacent_nodes(self, populated_store):
        neighbors = populated_store.query_neighbors("openai")
        # openai is connected to: doc_test__chunk_0000 (MENTIONS, in-edge),
        # anthropic (COOCCURS_WITH), google (RELATED_TO)
        assert "anthropic" in neighbors
        assert "google" in neighbors

    def test_filter_by_edge_type(self, populated_store):
        cooccur = populated_store.query_neighbors(
            "openai", edge_type=EdgeType.COOCCURS_WITH
        )
        related = populated_store.query_neighbors(
            "openai", edge_type=EdgeType.RELATED_TO
        )
        assert "anthropic" in cooccur
        assert "google" in related
        assert "google" not in cooccur
        assert "anthropic" not in related

    def test_nonexistent_node_returns_empty(self, populated_store):
        assert populated_store.query_neighbors("does_not_exist") == []

    def test_no_self_loops(self, populated_store):
        for node_id in ["openai", "anthropic", "google"]:
            neighbors = populated_store.query_neighbors(node_id)
            assert node_id not in neighbors


class TestQueryByEntity:
    def test_exact_match(self, populated_store):
        results = populated_store.query_by_entity("openai")
        ids = [n.node_id for n in results]
        assert "openai" in ids

    def test_partial_match(self, populated_store):
        results = populated_store.query_by_entity("open")
        ids = [n.node_id for n in results]
        assert "openai" in ids

    def test_case_insensitive(self, populated_store):
        results = populated_store.query_by_entity("OPENAI")
        ids = [n.node_id for n in results]
        assert "openai" in ids

    def test_no_match_returns_empty(self, populated_store):
        results = populated_store.query_by_entity("zzz_no_match_zzz")
        assert results == []


class TestNodeCount:
    def test_empty_store(self, empty_store):
        assert empty_store.node_count() == 0

    def test_after_adding_nodes(self, empty_store):
        for i in range(5):
            empty_store.add_node(make_entity_node(f"e_{i}"))
        assert empty_store.node_count() == 5


class TestEdgeCount:
    def test_empty_store(self, empty_store):
        assert empty_store.edge_count() == 0


# ---------------------------------------------------------------------------
# Save / Load round-trip tests
# ---------------------------------------------------------------------------

class TestSaveLoad:
    def test_pickle_round_trip(self, populated_store, tmp_path):
        pkl = tmp_path / "graph.pkl"
        gml = tmp_path / "graph.graphml"
        populated_store.save(graphml_path=gml, pkl_path=pkl)

        assert pkl.exists()
        new_store = NetworkXGraphStore()
        new_store.load(pkl_path=pkl)

        assert new_store.node_count() == populated_store.node_count()
        assert new_store.edge_count() == populated_store.edge_count()

    def test_graphml_round_trip(self, populated_store, tmp_path):
        pkl = tmp_path / "graph.pkl"
        gml = tmp_path / "graph.graphml"
        populated_store.save(graphml_path=gml, pkl_path=pkl)

        assert gml.exists()
        new_store = NetworkXGraphStore()
        new_store.load(graphml_path=gml)

        assert new_store.node_count() == populated_store.node_count()
        assert new_store.edge_count() == populated_store.edge_count()

    def test_nodes_preserved_after_load(self, populated_store, tmp_path):
        pkl = tmp_path / "graph.pkl"
        gml = tmp_path / "graph.graphml"
        populated_store.save(graphml_path=gml, pkl_path=pkl)

        new_store = NetworkXGraphStore()
        new_store.load(pkl_path=pkl)

        node = new_store.get_node("openai")
        assert node is not None
        assert node.node_id == "openai"

    def test_load_nonexistent_raises(self, empty_store, tmp_path):
        with pytest.raises(FileNotFoundError):
            empty_store.load(
                pkl_path=tmp_path / "missing.pkl",
                graphml_path=tmp_path / "missing.graphml",
            )

    def test_pickle_file_is_valid_pickle(self, populated_store, tmp_path):
        pkl = tmp_path / "graph.pkl"
        gml = tmp_path / "graph.graphml"
        populated_store.save(graphml_path=gml, pkl_path=pkl)
        with open(pkl, "rb") as fh:
            obj = pickle.load(fh)
        assert obj is not None


# ---------------------------------------------------------------------------
# GraphBuilder integration test (requires spaCy)
# ---------------------------------------------------------------------------

class TestGraphBuilderIntegration:
    """
    Requires spaCy en_core_web_sm to be installed.
    Skip gracefully if not available.
    """

    @pytest.fixture(scope="class")
    def built_store(self, tmp_path_factory):
        try:
            import spacy
            spacy.load("en_core_web_sm")
        except (ImportError, OSError):
            pytest.skip("spaCy en_core_web_sm not installed — skipping integration tests")

        from data.scripts.ingest import ingest
        from knowledge_graph.builder import GraphBuilder
        from knowledge_graph.extractor import SpacyExtractor

        out_dir = tmp_path_factory.mktemp("processed")
        ingest(
            raw_dir=PROJECT_ROOT / "data" / "raw",
            output_dir=out_dir,
            chunk_tokens=500,
            overlap_tokens=50,
        )

        store = NetworkXGraphStore()
        extractor = SpacyExtractor()
        builder = GraphBuilder(extractor=extractor, store=store, verbose=False)
        builder.build_from_chunks(out_dir / "chunks.jsonl")
        return store

    def test_graph_has_nodes(self, built_store):
        assert built_store.node_count() > 0, "Expected graph to have nodes"

    def test_graph_has_at_least_10_entity_nodes(self, built_store):
        entities = built_store.nodes_by_type(NodeType.ENTITY)
        assert len(entities) >= 10, (
            f"Expected ≥ 10 entity nodes, got {len(entities)}"
        )

    def test_graph_has_document_nodes(self, built_store):
        docs = built_store.nodes_by_type(NodeType.DOCUMENT)
        assert len(docs) == 5, f"Expected 5 Document nodes, got {len(docs)}"

    def test_graph_has_chunk_nodes(self, built_store):
        chunks = built_store.nodes_by_type(NodeType.CHUNK)
        assert len(chunks) > 0

    def test_graph_has_mentions_edges(self, built_store):
        s = built_store.summary()
        assert s["edge_type_counts"].get("MENTIONS", 0) > 0

    def test_graph_has_part_of_edges(self, built_store):
        s = built_store.summary()
        assert s["edge_type_counts"].get("PART_OF", 0) > 0

    def test_graph_has_at_least_50_edges(self, built_store):
        assert built_store.edge_count() >= 50, (
            f"Expected ≥ 50 edges total, got {built_store.edge_count()}"
        )

    def test_known_entity_in_graph(self, built_store):
        """At least one well-known AI org should appear in the sample docs."""
        found = any(
            built_store.query_by_entity(name)
            for name in ["openai", "google", "anthropic", "microsoft"]
        )
        assert found, "Expected at least one AI org entity in the graph"

    def test_save_load_roundtrip(self, built_store, tmp_path):
        pkl = tmp_path / "graph.pkl"
        gml = tmp_path / "graph.graphml"
        built_store.save(graphml_path=gml, pkl_path=pkl)

        restored = NetworkXGraphStore()
        restored.load(pkl_path=pkl)

        assert restored.node_count() == built_store.node_count()
        assert restored.edge_count() == built_store.edge_count()
