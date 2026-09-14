# Data + Knowledge Graph Interface Contract

**Module Owner**: Member 1 (M1)  
**Branch**: `feature/m1-data-kg`  
**Last Updated**: 2026-09-14  
**Status**: Stable — do not modify without coordination with M1

---

## Overview

This document defines the stable interfaces that downstream pipeline members (M2 – RAG/GraphRAG, M3 – Agents) should code against. All paths are relative to the project root.

---

## 1. Output File Locations

| Artifact | Path | Format | Description |
|---|---|---|---|
| Chunked documents | `data/processed/chunks.jsonl` | JSON Lines | One JSON object per line, one chunk per object |
| Knowledge graph (fast load) | `knowledge_graph/graph.pkl` | Python pickle | Primary runtime format — NetworkX `MultiDiGraph` |
| Knowledge graph (portable) | `data/processed/graph.graphml` | GraphML XML | Human-readable; use with Gephi, yEd, or graph tools |

> [!IMPORTANT]
> Always load from `graph.pkl` in code — it is faster and preserves full type information. Use `graph.graphml` for visualisation only.

---

## 2. Chunk JSONL Schema

Each line of `data/processed/chunks.jsonl` is a valid JSON object with the following fields:

| Field | Type | Example | Description |
|---|---|---|---|
| `doc_id` | `str` | `"doc1_llm_overview"` | Stable document identifier (filename stem, no extension) |
| `chunk_id` | `str` | `"doc1_llm_overview__chunk_0000"` | Unique chunk identifier: `<doc_id>__chunk_<NNNN>` |
| `source` | `str` | `"/abs/path/to/data/raw/doc1_llm_overview.txt"` | Absolute path to the source file |
| `text` | `str` | `"Large language models (LLMs) have …"` | Cleaned chunk text (~500 whitespace-split tokens) |
| `char_start` | `int` | `0` | Start character offset in the **cleaned** document text |
| `char_end` | `int` | `2841` | End character offset in the **cleaned** document text |
| `token_count` | `int` | `498` | Approximate whitespace-split token count |

### Reading chunks.jsonl

```python
import json
from pathlib import Path

chunks = []
with open("data/processed/chunks.jsonl", encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if line:
            chunks.append(json.loads(line))

# Access fields
for chunk in chunks:
    print(chunk["doc_id"], chunk["chunk_id"], chunk["token_count"])
    print(chunk["text"][:100])
```

---

## 3. GraphStore Interface

Import path: `from knowledge_graph.graph_store import NetworkXGraphStore, GraphStore`

### 3.1 Instantiation and Loading

```python
from knowledge_graph.graph_store import NetworkXGraphStore

store = NetworkXGraphStore()

# Load from pickle (preferred)
store.load(pkl_path="knowledge_graph/graph.pkl")

# Or load from GraphML
store.load(graphml_path="data/processed/graph.graphml")
```

### 3.2 Method Signatures

#### `add_node(node: NodeData) -> None`

Add or merge a node. If the node_id already exists, attributes are updated.

```python
from knowledge_graph.schema import NodeData, NodeType

store.add_node(NodeData(
    node_id="openai",
    node_type=NodeType.ENTITY,
    label="OpenAI",
    entity_label="ORG",
    source_doc_id="doc1_llm_overview",
))
```

#### `add_edge(edge: EdgeData) -> None`

Add or increment a directed edge. Weight is accumulated on duplicate edges.

```python
from knowledge_graph.schema import EdgeData, EdgeType

store.add_edge(EdgeData(
    source_id="doc1_llm_overview__chunk_0000",
    target_id="openai",
    edge_type=EdgeType.MENTIONS,
    weight=1.0,
    metadata={"chunk_id": "doc1_llm_overview__chunk_0000"},
))
```

#### `query_neighbors(node_id: str, edge_type: EdgeType | None = None) -> list[str]`

Return adjacent node_ids. Queries both successors and predecessors.

```python
from knowledge_graph.schema import EdgeType

# All neighbors
neighbors = store.query_neighbors("openai")
# → ["doc1_llm_overview__chunk_0000", "sam altman", "microsoft", ...]

# Only entities co-occurring with "openai"
cooccurring = store.query_neighbors("openai", edge_type=EdgeType.COOCCURS_WITH)
# → ["anthropic", "google deepmind", ...]
```

#### `query_by_entity(entity_text: str) -> list[NodeData]`

Case-insensitive partial-match search across all node labels and IDs.

```python
results = store.query_by_entity("openai")
for node in results:
    print(node.node_id, node.node_type, node.entity_label)
# → openai  NodeType.ENTITY  ORG
```

Returns `list[NodeData]` — see `knowledge_graph/schema.py` for the full `NodeData` definition.

#### `save(graphml_path, pkl_path) -> None`

```python
store.save(
    graphml_path="data/processed/graph.graphml",
    pkl_path="knowledge_graph/graph.pkl",
)
```

#### `load(graphml_path=None, pkl_path=None) -> None`

Pickle is loaded preferentially when both paths are supplied.

```python
store.load(pkl_path="knowledge_graph/graph.pkl")
```

#### `node_count() -> int` / `edge_count() -> int`

```python
print(store.node_count())   # e.g. 312
print(store.edge_count())   # e.g. 874
```

#### `summary() -> dict`

```python
s = store.summary()
# {
#   "total_nodes": 312,
#   "total_edges": 874,
#   "node_type_counts": {"Entity": 280, "Document": 5, "Chunk": 27},
#   "edge_type_counts": {"MENTIONS": 270, "PART_OF": 27, "COOCCURS_WITH": 540, "RELATED_TO": 37}
# }
```

---

## 4. Schema Reference

Import path: `from knowledge_graph.schema import NodeType, EdgeType, NodeData, EdgeData`

### NodeType (enum)

| Value | Description |
|---|---|
| `NodeType.ENTITY` | A named entity (person, org, location, etc.) |
| `NodeType.DOCUMENT` | A source document |
| `NodeType.CHUNK` | A text chunk derived from a document |

### EdgeType (enum)

| Value | Direction | Description |
|---|---|---|
| `EdgeType.MENTIONS` | Chunk → Entity | Chunk contains a mention of the entity |
| `EdgeType.PART_OF` | Chunk → Document | Chunk belongs to the document |
| `EdgeType.COOCCURS_WITH` | Entity ↔ Entity | Both entities appear in the same chunk |
| `EdgeType.RELATED_TO` | Entity ↔ Entity | Both entities appear in the same document (cross-chunk) |

### NodeData fields

| Field | Type | Description |
|---|---|---|
| `node_id` | `str` | Unique node identifier |
| `node_type` | `NodeType` | Node category |
| `label` | `str` | Human-readable label |
| `entity_label` | `str \| None` | NER label for Entity nodes (e.g. "ORG", "PERSON") |
| `source_doc_id` | `str \| None` | doc_id of the originating document |
| `metadata` | `dict` | Arbitrary extra attributes |

---

## 5. Running the Full Pipeline

```bash
# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run end-to-end pipeline from project root
python knowledge_graph/build.py

# With custom settings
python knowledge_graph/build.py \
    --raw-dir data/raw \
    --output-dir data/processed \
    --chunk-tokens 500 \
    --overlap-tokens 50
```

---

## 6. End-to-End Usage Example (for M2 / M3)

```python
from knowledge_graph.graph_store import NetworkXGraphStore
from knowledge_graph.schema import EdgeType, NodeType

# 1. Load the pre-built graph
store = NetworkXGraphStore()
store.load(pkl_path="knowledge_graph/graph.pkl")

print(f"Graph has {store.node_count()} nodes and {store.edge_count()} edges")

# 2. Look up an entity
matches = store.query_by_entity("transformer")
for node in matches:
    print(node.node_id, node.entity_label)

# 3. Get neighbors (for GraphRAG context retrieval)
neighbors = store.query_neighbors("transformer", edge_type=EdgeType.COOCCURS_WITH)
print("Co-occurs with:", neighbors[:5])

# 4. Load chunks for a given doc
import json
chunks = [
    json.loads(l) for l in open("data/processed/chunks.jsonl")
    if json.loads(l)["doc_id"] == "doc1_llm_overview"
]
```

---

## 7. Adding Support for New Extractors

To swap in an LLM-based extractor:

1. Open `knowledge_graph/extractor.py`
2. Implement `LLMExtractor(BaseExtractor)` (the stub is already there)
3. Pass it to `GraphBuilder`:

```python
from knowledge_graph.extractor import LLMExtractor
from knowledge_graph.builder import GraphBuilder
from knowledge_graph.graph_store import NetworkXGraphStore

extractor = LLMExtractor(model="gpt-4o-mini")
store = NetworkXGraphStore()
builder = GraphBuilder(extractor=extractor, store=store)
builder.build_from_chunks("data/processed/chunks.jsonl")
```

No other files need to change.
