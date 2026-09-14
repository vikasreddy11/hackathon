# RAG & GraphRAG Interface Contract

**Module Owner**: Member 2 (M2)  
**Branch**: `feature/m2-rag-graphrag`  
**Last Updated**: 2026-09-14  
**Status**: Stable — primary interface for M3 (Agents) and M4 (Evaluation)

---

## Overview

This document specifies the interface contracts for the standard **RAG** and **GraphRAG** retrieval-augmented generation pipelines.
Both pipelines provide standard, predictable function signatures and return schemas so Member 3 (Agents) can orchestrate them and Member 4 (Evaluation) can assess them side-by-side.

All paths referenced are relative to the repository root.

---

## 1. Quick Import Reference

```python
# Pipeline entrypoints
from rag import run_rag
from graphrag import run_graphrag

# Low-level RAG components
from rag import RAGRetriever, VectorStoreIndexer, generate_answer
from rag.llm_client import LLMClient, call_llm

# Low-level GraphRAG components
from graphrag import GraphRetriever, EntityExtractor, build_graph_context, generate_graphrag_answer
```

---

## 2. Pipeline 1: Standard RAG Interface

### 2.1 Function Signature

```python
def run_rag(
    question: str,
    top_k: int = 3,
    backend: str | None = None,
    model: str | None = None,
    index_dir: str | Path = "rag/vector_store",
    retriever: RAGRetriever | None = None,
    **kwargs,
) -> dict[str, Any]
```

### 2.2 Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | `str` | *required* | The question or prompt to answer |
| `top_k` | `int` | `3` | Number of most similar document chunks to retrieve |
| `backend` | `str \| None` | `None` | LLM backend: `'auto'`, `'mock'`, `'groq'`, `'openai'`, `'gemini'`, `'anthropic'` |
| `model` | `str \| None` | `None` | Specific model identifier override for the backend |
| `index_dir` | `str \| Path` | `"rag/vector_store"` | Directory containing `index.faiss` and `metadata.json` |
| `retriever` | `RAGRetriever \| None` | `None` | Optional pre-initialized `RAGRetriever` instance for fast reuse |

### 2.3 Return Shape

```python
{
    "answer": str,       # Synthesized answer string from LLM
    "sources": list,     # List of retrieved chunk dictionaries
    "scores": list,      # List of cosine similarity scores (float, between 0.0 and 1.0)
}
```

#### Detailed `sources` Element Schema

Each item in `sources` has the following fields:

```python
{
    "chunk_id": str,     # E.g. "doc1_llm_overview__chunk_0000"
    "doc_id": str,       # E.g. "doc1_llm_overview"
    "text": str,         # Clean text content of the retrieved chunk
    "source": str,       # Path to original source document
    "score": float,      # Inner product / cosine similarity score
}
```

### 2.4 Example Usage

```python
from rag import run_rag

result = run_rag("What are the key benefits of Retrieval-Augmented Generation?", top_k=3)

print("Answer:", result["answer"])
for src, score in zip(result["sources"], result["scores"]):
    print(f"[{score:.3f}] {src['chunk_id']} from {src['doc_id']}")
```

---

## 3. Pipeline 2: GraphRAG Interface

### 3.1 Function Signature

```python
def run_graphrag(
    question: str,
    top_k_entities: int = 10,
    max_chunks: int = 5,
    backend: str | None = None,
    model: str | None = None,
    pkl_path: str | Path = "knowledge_graph/graph.pkl",
    chunks_path: str | Path = "data/processed/chunks.jsonl",
    retriever: GraphRetriever | None = None,
    extractor: EntityExtractor | None = None,
    **kwargs,
) -> dict[str, Any]
```

### 3.2 Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `question` | `str` | *required* | The question or prompt to answer |
| `top_k_entities` | `int` | `10` | Maximum entities to discover in the subgraph expansion |
| `max_chunks` | `int` | `5` | Maximum supporting document chunks to attach to the context |
| `backend` | `str \| None` | `None` | LLM backend: `'auto'`, `'mock'`, `'groq'`, `'openai'`, `'gemini'`, `'anthropic'` |
| `model` | `str \| None` | `None` | Model identifier override |
| `pkl_path` | `str \| Path` | `"knowledge_graph/graph.pkl"` | Path to pre-built knowledge graph pickle |
| `chunks_path` | `str \| Path` | `"data/processed/chunks.jsonl"` | Path to document chunks JSONL |
| `retriever` | `GraphRetriever \| None` | `None` | Optional pre-initialized `GraphRetriever` |
| `extractor` | `EntityExtractor \| None` | `None` | Optional pre-initialized `EntityExtractor` |

### 3.3 Return Shape

```python
{
    "answer": str,       # Synthesized answer string from LLM
    "sources": list,     # List of supporting chunk dictionaries
    "subgraph": dict,    # Retrieved graph topology and components
}
```

#### Detailed `subgraph` Schema

```python
{
    "query_entities": list[str],      # Extracted entity phrases from the query
    "entities": [                     # Discovered entity nodes
        {
            "node_id": str,           # Unique entity identifier (lowercased)
            "label": str,             # Entity surface text
            "entity_label": str,      # NER type: "ORG", "PERSON", "PRODUCT", etc.
            "source_doc_id": str,     # Originating doc_id
            "is_seed": bool,          # True if directly matched query, False if 1-hop neighbor
        },
        ...
    ],
    "relationships": [                # Inter-entity and entity-chunk edges
        {
            "source": str,            # Source node_id
            "target": str,            # Target node_id
            "edge_type": str,         # "COOCCURS_WITH", "RELATED_TO", "MENTIONS", etc.
            "weight": float,          # Relationship weight
            "metadata": dict,         # Extra edge metadata
        },
        ...
    ],
    "chunk_ids": list[str],           # Unique chunk IDs connected to retrieved entities
}
```

#### Detailed `sources` Element Schema

Each item in `sources` is structured consistently with RAG:

```python
{
    "chunk_id": str,     # Unique chunk ID
    "doc_id": str,       # Source document identifier
    "text": str,         # Clean text content of the supporting chunk
    "source": str,       # Path to original source file
}
```

### 3.4 Example Usage

```python
from graphrag import run_graphrag

result = run_graphrag("How do transformers connect to attention mechanisms?")

print("Answer:\n", result["answer"])
print("Matched query entities:", result["subgraph"]["query_entities"])
print("Total entities in subgraph:", len(result["subgraph"]["entities"]))
print("Total relationships:", len(result["subgraph"]["relationships"]))
for rel in result["subgraph"]["relationships"][:5]:
    print(f"  {rel['source']} --[{rel['edge_type']}]--> {rel['target']}")
```

---

## 4. LLM Backend Configuration

Both pipelines utilize the swappable `LLMClient` in `rag/llm_client.py`.
By default (`backend="auto"`), it inspects environment variables in this priority order:

1. `GROQ_API_KEY` (using `llama-3.1-8b-instant`)
2. `OPENAI_API_KEY` (using `gpt-4o-mini`)
3. `GEMINI_API_KEY` (using `gemini-2.0-flash`)
4. `ANTHROPIC_API_KEY` (using `claude-3-5-haiku-20241022`)
5. If no remote API keys are set, it falls back to a deterministic, offline extractive synthesis mock.

This ensures all unit tests, automated CI checks, and local experiments run with **100% reliability and zero credentials**.

---

## 5. Artifact File Locations

| Pipeline | Artifact | Default Path | Format |
|---|---|---|---|
| RAG | Vector Index | `rag/vector_store/index.faiss` | FAISS IndexFlatIP (cosine) |
| RAG | Vector Metadata | `rag/vector_store/metadata.json` | JSON mapping vector IDs to chunk data |
| GraphRAG | Source Graph | `knowledge_graph/graph.pkl` | NetworkX MultiDiGraph via GraphStore |
| GraphRAG | Source Chunks | `data/processed/chunks.jsonl` | Chunks JSON Lines |

To initialize the vector store manually:
```bash
python -m rag.embed --chunks data/processed/chunks.jsonl --out rag/vector_store
```
*(Note: `RAGRetriever` will auto-build the vector index if `chunks.jsonl` exists and the index is missing).*
