"""
agents/tools.py
===============
The five tools available to the agentic orchestrator.

What is a "tool"?
-----------------
A tool is just a function the agent can CHOOSE to call.
Each tool wraps an existing pipeline capability (RAG retriever, graph retriever, etc.)
and returns a standardized ToolResult so the agent can record it in its state.

The five tools
--------------
1. vector_search   — similarity search in FAISS vector index (wraps RAGRetriever)
2. entity_search   — find entities in the knowledge graph that match query terms
3. graph_search    — expand a set of entity IDs to their 1-hop neighborhood
4. document_search — retrieve full text of specific document chunks by ID
5. evaluate_evidence — ask Gemini: "Is this evidence enough to answer the question?"

How the agent uses them
-----------------------
The orchestrator asks Gemini: "Given what you know so far, which tool should I call
and why?" Gemini responds with a tool name + arguments. The orchestrator calls the
corresponding function below, records the ToolResult, and updates AgentState.
"""

from __future__ import annotations

import re
from typing import Any

from agents.state import ToolResult
from graphrag.entity_extractor import EntityExtractor
from graphrag.graph_retriever import GraphRetriever
from knowledge_graph.graph_store import NetworkXGraphStore
from rag.llm_client import call_llm
from rag.retriever import RAGRetriever


# ---------------------------------------------------------------------------
# Singletons — loaded once, reused across all tool calls in a session
# ---------------------------------------------------------------------------

_rag_retriever: RAGRetriever | None = None
_graph_retriever: GraphRetriever | None = None
_entity_extractor: EntityExtractor | None = None


def _get_rag_retriever() -> RAGRetriever:
    global _rag_retriever
    if _rag_retriever is None:
        _rag_retriever = RAGRetriever()
        _rag_retriever.load_or_initialize()
    return _rag_retriever


def _get_graph_retriever() -> GraphRetriever:
    global _graph_retriever
    if _graph_retriever is None:
        _graph_retriever = GraphRetriever()
    return _graph_retriever


def _get_entity_extractor() -> EntityExtractor:
    global _entity_extractor
    if _entity_extractor is None:
        _entity_extractor = EntityExtractor()
    return _entity_extractor


def _approx_tokens(text: str) -> int:
    """Approximate token count using whitespace splitting."""
    return len(text.split())


# ===========================================================================
# Tool 1 — vector_search
# ===========================================================================

def vector_search(
    query: str,
    top_k: int = 3,
    reasoning: str = "",
) -> ToolResult:
    """
    Embed the query and search the FAISS vector index for similar document chunks.

    When to use this tool
    ---------------------
    - The question is broad or conceptual and we need relevant text passages.
    - We don't yet know which entities are involved.
    - We want to quickly get grounding text before exploring the graph.

    Parameters
    ----------
    query : str
        The question or search phrase to embed.
    top_k : int
        Number of most similar chunks to return (default 3).
    reasoning : str
        Why the orchestrator chose this tool (for the trace).

    Returns
    -------
    ToolResult with outputs = list of chunk dicts (chunk_id, doc_id, text, score).
    """
    retriever = _get_rag_retriever()
    chunks = retriever.retrieve(query=query, top_k=top_k)
    tokens = _approx_tokens(query) + sum(_approx_tokens(c.get("text", "")) for c in chunks)

    return ToolResult(
        tool_name="vector_search",
        inputs={"query": query, "top_k": top_k},
        outputs=chunks,
        tokens_used=tokens,
        reasoning=reasoning,
    )


# ===========================================================================
# Tool 2 — entity_search
# ===========================================================================

def entity_search(
    query: str,
    known_entities: list[str] | None = None,
    max_entities: int = 10,
    max_chunks: int = 3,
    reasoning: str = "",
) -> ToolResult:
    """
    Extract entities from the query and look them up in the knowledge graph.
    Returns matched entity nodes, their 1-hop neighbors, and linked chunks.

    When to use this tool
    ---------------------
    - We know specific names/concepts are important (e.g. "OpenAI", "RLHF").
    - We want to find what entities the knowledge graph knows about.
    - We need to seed the graph traversal.

    Parameters
    ----------
    query : str
        The question or search phrase to extract entities from.
    known_entities : list[str] | None
        Optional pre-known entity IDs to look up directly.
    max_entities : int
        Max entity nodes to include in the result.
    max_chunks : int
        Max source chunks to attach to result.
    reasoning : str
        Why the orchestrator chose this tool.

    Returns
    -------
    ToolResult with outputs = subgraph dict:
        {query_entities, entities, relationships, chunk_ids, chunks}
    """
    extractor = _get_entity_extractor()
    graph_retriever = _get_graph_retriever()

    # Extract entity terms from the query
    entity_terms = extractor.extract_entities(query, known_entities=known_entities)

    # Retrieve matching subgraph
    subgraph = graph_retriever.retrieve_subgraph(
        entity_queries=entity_terms,
        max_entities=max_entities,
        max_chunks=max_chunks,
    )

    tokens = _approx_tokens(query)
    return ToolResult(
        tool_name="entity_search",
        inputs={"query": query, "known_entities": known_entities, "max_entities": max_entities},
        outputs=subgraph,
        tokens_used=tokens,
        reasoning=reasoning,
    )


# ===========================================================================
# Tool 3 — graph_search
# ===========================================================================

def graph_search(
    entity_ids: list[str],
    max_entities: int = 15,
    max_chunks: int = 4,
    reasoning: str = "",
) -> ToolResult:
    """
    Expand a set of known entity IDs in the knowledge graph to their 1-hop
    neighborhood — discovering related entities, relationships, and linked chunks.

    When to use this tool
    ---------------------
    - We already found some entities (via entity_search) and want to go deeper.
    - We suspect related entities are relevant but were not in the original query.
    - Multi-hop reasoning: "A is connected to B, what is B connected to?"

    Parameters
    ----------
    entity_ids : list[str]
        Node IDs (lowercased entity names) to expand in the graph.
    max_entities : int
        Max entity nodes in the expanded result.
    max_chunks : int
        Max source chunks to return.
    reasoning : str
        Why the orchestrator chose this tool.

    Returns
    -------
    ToolResult with outputs = subgraph dict:
        {query_entities, entities, relationships, chunk_ids, chunks}
    """
    graph_retriever = _get_graph_retriever()
    subgraph = graph_retriever.retrieve_subgraph(
        entity_queries=entity_ids,
        max_entities=max_entities,
        max_chunks=max_chunks,
    )

    return ToolResult(
        tool_name="graph_search",
        inputs={"entity_ids": entity_ids, "max_entities": max_entities},
        outputs=subgraph,
        tokens_used=0,
        reasoning=reasoning,
    )


# ===========================================================================
# Tool 4 — document_search
# ===========================================================================

def document_search(
    chunk_ids: list[str],
    reasoning: str = "",
) -> ToolResult:
    """
    Retrieve full text of specific document chunks by their chunk_id.

    When to use this tool
    ---------------------
    - We know specific chunk IDs (from graph_search or entity_search results)
      but don't yet have their full text.
    - We want to verify or expand on a specific piece of evidence.

    Parameters
    ----------
    chunk_ids : list[str]
        List of chunk_id strings to retrieve full text for.
    reasoning : str
        Why the orchestrator chose this tool.

    Returns
    -------
    ToolResult with outputs = list of chunk dicts (chunk_id, doc_id, text, source).
    """
    graph_retriever = _get_graph_retriever()
    chunks = [graph_retriever.get_chunk_data(cid) for cid in chunk_ids]
    # Filter out empty results
    chunks = [c for c in chunks if c.get("text")]

    tokens = sum(_approx_tokens(c.get("text", "")) for c in chunks)
    return ToolResult(
        tool_name="document_search",
        inputs={"chunk_ids": chunk_ids},
        outputs=chunks,
        tokens_used=tokens,
        reasoning=reasoning,
    )


# ===========================================================================
# Tool 5 — evaluate_evidence
# ===========================================================================

_EVALUATE_SYSTEM_PROMPT = (
    "You are an expert evidence evaluator. Your job is to assess whether the "
    "collected evidence is sufficient to accurately and completely answer the question. "
    "Be critical and precise. Respond ONLY with valid JSON."
)

_EVALUATE_PROMPT_TEMPLATE = """\
Question: {question}

Evidence collected so far:
{evidence_text}

Evaluate the evidence and respond with this JSON structure ONLY:
{{
  "score": <integer 0-10, where 10 = fully sufficient to answer accurately>,
  "ready": <true if score >= 7, false otherwise>,
  "missing": [<list of strings describing what key information is still missing>],
  "reasoning": "<one sentence explaining the score>"
}}

Score guide:
- 0-3: Evidence is very sparse, question cannot be answered
- 4-6: Partial evidence, answer would be incomplete or uncertain  
- 7-8: Good evidence, answer can be given with reasonable confidence
- 9-10: Excellent evidence, answer can be given with high confidence
"""


def evaluate_evidence(
    question: str,
    evidence: list[dict[str, Any]],
    backend: str | None = "gemini",
    model: str | None = None,
    reasoning: str = "",
) -> ToolResult:
    """
    Ask Gemini to evaluate whether the current evidence is sufficient to
    answer the question. Returns a score, readiness flag, and what is missing.

    When to use this tool
    ---------------------
    - After collecting some evidence and before deciding to stop or continue.
    - The orchestrator uses this to decide: "Do I have enough? Or should I search more?"

    Parameters
    ----------
    question : str
        The original user question.
    evidence : list[dict]
        All chunks retrieved so far (from AgentState.evidence).
    backend : str | None
        LLM backend to use (default: "gemini").
    model : str | None
        LLM model override.
    reasoning : str
        Why the orchestrator chose this tool.

    Returns
    -------
    ToolResult with outputs = dict:
        {"score": int, "ready": bool, "missing": list[str], "reasoning": str}
    """
    # Build evidence text (cap at 5 chunks to keep prompt manageable)
    evidence_lines = []
    for i, chunk in enumerate(evidence[:5], 1):
        cid = chunk.get("chunk_id", f"chunk_{i}")
        text = chunk.get("text", "")[:400].replace("\n", " ")
        evidence_lines.append(f"[{i}] {cid}: {text}...")

    if not evidence_lines:
        evidence_text = "No evidence has been retrieved yet."
    else:
        evidence_text = "\n".join(evidence_lines)
        if len(evidence) > 5:
            evidence_text += f"\n... and {len(evidence) - 5} more chunks."

    prompt = _EVALUATE_PROMPT_TEMPLATE.format(
        question=question,
        evidence_text=evidence_text,
    )

    tokens_in = _approx_tokens(prompt)

    try:
        raw_response = call_llm(
            prompt=prompt,
            system_prompt=_EVALUATE_SYSTEM_PROMPT,
            backend=backend,
            model=model,
        )
        result = _parse_eval_json(raw_response)
    except Exception as exc:
        # Graceful fallback: if parsing fails, assume not ready
        result = {
            "score": 4,
            "ready": False,
            "missing": [f"Evaluation failed: {exc}"],
            "reasoning": "Could not parse LLM evaluation response.",
        }

    tokens_out = _approx_tokens(str(result))
    return ToolResult(
        tool_name="evaluate_evidence",
        inputs={"question": question, "evidence_count": len(evidence)},
        outputs=result,
        tokens_used=tokens_in + tokens_out,
        reasoning=reasoning,
    )


def _parse_eval_json(text: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response, handling markdown code fences."""
    # Strip markdown code fences if present
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

    import json
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object within response
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            raise ValueError(f"No valid JSON found in: {text[:200]}")

    return {
        "score": int(data.get("score", 5)),
        "ready": bool(data.get("ready", False)),
        "missing": list(data.get("missing", [])),
        "reasoning": str(data.get("reasoning", "")),
    }


# ===========================================================================
# Tool registry — maps tool name strings to callable functions
# ===========================================================================

TOOL_REGISTRY: dict[str, Any] = {
    "vector_search": vector_search,
    "entity_search": entity_search,
    "graph_search": graph_search,
    "document_search": document_search,
    "evaluate_evidence": evaluate_evidence,
}

TOOL_DESCRIPTIONS = {
    "vector_search": (
        "Search document chunks by semantic similarity. Use for broad or conceptual questions "
        "when you need relevant text passages. Args: query (str), top_k (int, default 3)."
    ),
    "entity_search": (
        "Find entities in the knowledge graph matching the query. Use when specific named entities "
        "are important. Returns entities, relationships, and linked chunks. Args: query (str)."
    ),
    "graph_search": (
        "Expand known entity IDs to their 1-hop graph neighborhood. Use for multi-hop reasoning "
        "after entity_search. Args: entity_ids (list of entity node_id strings)."
    ),
    "document_search": (
        "Retrieve full text of specific chunks by chunk_id. Use when you have chunk IDs from "
        "graph results but need the full text. Args: chunk_ids (list of strings)."
    ),
    "evaluate_evidence": (
        "Evaluate whether collected evidence is sufficient to answer the question. Returns a score "
        "0-10 and what is still missing. Use before deciding to stop or continue. No extra args needed."
    ),
}
