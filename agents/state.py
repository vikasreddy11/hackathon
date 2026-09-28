"""
agents/state.py
===============
Core data structures for the agentic investigation loop.

Terminology for beginners
--------------------------
- AgentState : a "notebook" the agent carries through the whole investigation.
  Every tool result, every entity found, every reasoning step is recorded here.
- ToolResult : one tool call's input + output + reasoning, recorded as a trace step.

Nothing in this file talks to an LLM or database — it is pure data storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# ToolResult — one step in the investigation
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """
    Records a single tool invocation.

    Fields
    ------
    tool_name : str
        Name of the tool called (e.g. "vector_search", "graph_search").
    inputs : dict
        The arguments passed to the tool.
    outputs : Any
        Whatever the tool returned.
    tokens_used : int
        Approximate tokens used by this step (0 for non-LLM tools).
    reasoning : str
        The orchestrator's explanation of WHY it called this tool.
    step_number : int
        Which step in the investigation sequence this is (1-indexed).
    """
    tool_name: str
    inputs: dict[str, Any]
    outputs: Any
    tokens_used: int = 0
    reasoning: str = ""
    step_number: int = 0


# ---------------------------------------------------------------------------
# AgentState — the agent's full investigation notebook
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    """
    The complete state of an ongoing agent investigation.

    Think of this as the agent's "notebook" — everything it has discovered,
    every step it has taken, and its current evidence is stored here.

    Fields
    ------
    question : str
        The original user question being investigated.
    evidence : list[dict]
        All retrieved document chunks accumulated across tool calls.
        Each chunk has: chunk_id, doc_id, text, source, score.
    entities_found : list[str]
        All entity strings discovered during the investigation.
    subgraph_data : dict
        Merged graph data (entities, relationships, chunk_ids) from graph tools.
    trace : list[ToolResult]
        The full step-by-step record of every tool call made.
    step_count : int
        How many tool calls have been made so far.
    total_tokens_approx : int
        Running total of approximate tokens used (for budget tracking).
    stopped_reason : str
        Why the agent stopped. One of:
        "sufficient_evidence", "max_steps", "budget_exceeded", "no_progress".
    final_answer : str
        The LLM-generated answer produced after investigation ends.
    final_sources : list[dict]
        De-duplicated list of source chunks cited in the final answer.
    """
    question: str

    # Accumulated evidence
    evidence: list[dict[str, Any]] = field(default_factory=list)
    entities_found: list[str] = field(default_factory=list)
    subgraph_data: dict[str, Any] = field(default_factory=lambda: {
        "entities": [],
        "relationships": [],
        "chunk_ids": [],
        "chunks": [],
    })

    # Investigation tracking
    trace: list[ToolResult] = field(default_factory=list)
    step_count: int = 0
    total_tokens_approx: int = 0
    strategy_changes: int = 0
    latest_score: int = 0
    latest_ready: bool = False
    latest_missing: list[str] = field(default_factory=list)

    # Outcome
    stopped_reason: str = ""
    final_answer: str = ""
    final_sources: list[dict[str, Any]] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------------

    def add_tool_result(self, result: ToolResult) -> None:
        """
        Record a tool result in the trace and update running totals.
        Also merges any chunks or entities from the result into the state.
        """
        result.step_number = self.step_count + 1
        self.trace.append(result)
        self.step_count += 1
        self.total_tokens_approx += result.tokens_used

        # Merge chunks into evidence (de-duplicate by chunk_id)
        existing_chunk_ids = {c.get("chunk_id") for c in self.evidence}
        outputs = result.outputs

        if isinstance(outputs, list):
            # Tool returned a list of chunks (e.g. vector_search)
            for chunk in outputs:
                cid = chunk.get("chunk_id", "")
                if cid and cid not in existing_chunk_ids:
                    self.evidence.append(chunk)
                    existing_chunk_ids.add(cid)

        elif isinstance(outputs, dict):
            # Tool returned a subgraph dict (e.g. entity_search, graph_search)
            self._merge_subgraph(outputs)

    def _merge_subgraph(self, subgraph: dict[str, Any]) -> None:
        """Merge a subgraph dict into the accumulated subgraph_data."""
        # Merge entities
        existing_eids = {e["node_id"] for e in self.subgraph_data["entities"]}
        for ent in subgraph.get("entities", []):
            if ent.get("node_id") not in existing_eids:
                self.subgraph_data["entities"].append(ent)
                existing_eids.add(ent["node_id"])

        # Merge entity strings into entities_found
        for ent_str in subgraph.get("query_entities", []):
            if ent_str not in self.entities_found:
                self.entities_found.append(ent_str)

        # Merge relationships (de-dup by source+target+type)
        existing_rels = {
            (r["source"], r["target"], r["edge_type"])
            for r in self.subgraph_data["relationships"]
        }
        for rel in subgraph.get("relationships", []):
            key = (rel.get("source"), rel.get("target"), rel.get("edge_type"))
            if key not in existing_rels:
                self.subgraph_data["relationships"].append(rel)
                existing_rels.add(key)

        # Merge chunk_ids
        existing_cids = set(self.subgraph_data["chunk_ids"])
        for cid in subgraph.get("chunk_ids", []):
            if cid not in existing_cids:
                self.subgraph_data["chunk_ids"].append(cid)
                existing_cids.add(cid)

        # Merge chunks into evidence
        existing_evidence_ids = {c.get("chunk_id") for c in self.evidence}
        for chunk in subgraph.get("chunks", []):
            cid = chunk.get("chunk_id", "")
            if cid and cid not in existing_evidence_ids:
                self.evidence.append(chunk)
                existing_evidence_ids.add(cid)

    def get_evidence_summary(self) -> str:
        """
        Return a compact text summary of evidence collected so far.
        Used by the orchestrator to build its decision prompt.
        """
        if not self.evidence:
            return "No evidence retrieved yet."

        lines = []
        for i, chunk in enumerate(self.evidence[:5], 1):  # cap at 5 for prompt brevity
            cid = chunk.get("chunk_id", f"chunk_{i}")
            doc = chunk.get("doc_id", "unknown")
            text_preview = chunk.get("text", "")[:200].replace("\n", " ")
            lines.append(f"[{i}] {cid} (from {doc}): {text_preview}...")

        summary = "\n".join(lines)
        total = len(self.evidence)
        if total > 5:
            summary += f"\n... and {total - 5} more chunks retrieved."
        return summary

    def get_trace_summary(self) -> str:
        """
        Return a compact text summary of steps taken so far.
        Used by the orchestrator to avoid repeating itself.
        """
        if not self.trace:
            return "No steps taken yet."
        lines = []
        for step in self.trace:
            lines.append(
                f"Step {step.step_number}: [{step.tool_name}] — {step.reasoning[:100]}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full state to a plain dict (for JSON output)."""
        return {
            "question": self.question,
            "step_count": self.step_count,
            "total_tokens_approx": self.total_tokens_approx,
            "stopped_reason": self.stopped_reason,
            "final_answer": self.final_answer,
            "entities_found": self.entities_found,
            "evidence_count": len(self.evidence),
            "final_sources": self.final_sources,
            "trace": [
                {
                    "step": r.step_number,
                    "tool": r.tool_name,
                    "reasoning": r.reasoning,
                    "inputs": r.inputs,
                    "tokens": r.tokens_used,
                    "output_summary": _summarize_output(r.outputs),
                }
                for r in self.trace
            ],
            "subgraph": {
                "entity_count": len(self.subgraph_data["entities"]),
                "relationship_count": len(self.subgraph_data["relationships"]),
                "chunk_ids": self.subgraph_data["chunk_ids"],
            },
        }


def _summarize_output(outputs: Any) -> str:
    """Create a brief string summary of tool outputs for the trace."""
    if outputs is None:
        return "None"
    if isinstance(outputs, list):
        return f"List of {len(outputs)} items"
    if isinstance(outputs, dict):
        keys = list(outputs.keys())[:4]
        return f"Dict with keys: {keys}"
    if isinstance(outputs, str):
        return outputs[:120]
    return str(outputs)[:120]
