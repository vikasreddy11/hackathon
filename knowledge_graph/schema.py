"""
knowledge_graph/schema.py
=========================
Graph schema definitions: node types, edge types, and associated dataclasses.
These are the stable vocabulary shared across the builder, extractor, and
graph_store layers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Node and Edge type enumerations
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """Supported node types in the knowledge graph."""
    ENTITY = "Entity"        # A named entity extracted from text (person, org, …)
    DOCUMENT = "Document"   # A source document
    CHUNK = "Chunk"          # A text chunk derived from a document


class EdgeType(str, Enum):
    """Supported edge (relationship) types in the knowledge graph."""
    MENTIONS = "MENTIONS"            # Chunk → Entity  (chunk mentions entity)
    PART_OF = "PART_OF"              # Chunk → Document (chunk is part of document)
    RELATED_TO = "RELATED_TO"        # Entity ↔ Entity  (cross-chunk, same document)
    COOCCURS_WITH = "COOCCURS_WITH"  # Entity ↔ Entity  (within the same chunk)


# ---------------------------------------------------------------------------
# Entity type labels (matches spaCy's en_core_web_sm labels + extras)
# ---------------------------------------------------------------------------

class EntityLabel(str, Enum):
    """Common NER entity labels."""
    PERSON = "PERSON"
    ORG = "ORG"
    GPE = "GPE"               # Geopolitical entity (country, city, …)
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    DATE = "DATE"
    NORP = "NORP"             # Nationalities, religious groups, etc.
    FAC = "FAC"               # Buildings, airports, etc.
    LOC = "LOC"               # Non-GPE locations
    MONEY = "MONEY"
    PERCENT = "PERCENT"
    TIME = "TIME"
    QUANTITY = "QUANTITY"
    ORDINAL = "ORDINAL"
    CARDINAL = "CARDINAL"
    MISC = "MISC"             # Catch-all for other recognised entities
    UNKNOWN = "UNKNOWN"


# ---------------------------------------------------------------------------
# Node dataclass
# ---------------------------------------------------------------------------

@dataclass
class NodeData:
    """
    Attributes attached to every node in the graph.

    Fields
    ------
    node_id : str
        Unique identifier for this node.
        - Entity nodes  : normalised entity text (lowercased), e.g. "openai"
        - Document nodes: doc_id from chunks.jsonl
        - Chunk nodes   : chunk_id from chunks.jsonl
    node_type : NodeType
        Discriminator for the kind of node.
    label : str
        Human-readable label (entity text, doc filename, chunk ID).
    entity_label : str | None
        For Entity nodes: the NER label (e.g. "ORG", "PERSON").
    source_doc_id : str | None
        The document this node originates from (for Chunk & Entity nodes).
    metadata : dict
        Arbitrary extra metadata (e.g. char offsets, token count).
    """
    node_id: str
    node_type: NodeType
    label: str
    entity_label: str | None = None
    source_doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label,
            "entity_label": self.entity_label or "",
            "source_doc_id": self.source_doc_id or "",
            **{f"meta_{k}": str(v) for k, v in self.metadata.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NodeData":
        meta = {
            k[5:]: v for k, v in d.items() if k.startswith("meta_")
        }
        return cls(
            node_id=d["node_id"],
            node_type=NodeType(d["node_type"]),
            label=d["label"],
            entity_label=d.get("entity_label") or None,
            source_doc_id=d.get("source_doc_id") or None,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# Edge dataclass
# ---------------------------------------------------------------------------

@dataclass
class EdgeData:
    """
    Attributes attached to every edge in the graph.

    Fields
    ------
    source_id : str
        node_id of the source node.
    target_id : str
        node_id of the target node.
    edge_type : EdgeType
        Semantic type of the relationship.
    weight : float
        Co-occurrence or confidence weight (default 1.0).
    metadata : dict
        Arbitrary extra metadata (e.g. chunk_id where relation was found).
    """
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type.value,
            "weight": self.weight,
            **{f"meta_{k}": str(v) for k, v in self.metadata.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EdgeData":
        meta = {
            k[5:]: v for k, v in d.items() if k.startswith("meta_")
        }
        return cls(
            source_id=d["source_id"],
            target_id=d["target_id"],
            edge_type=EdgeType(d["edge_type"]),
            weight=float(d.get("weight", 1.0)),
            metadata=meta,
        )
