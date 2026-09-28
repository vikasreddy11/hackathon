"""
agents/pipeline.py
==================
Public entrypoint for Pipeline 3 — Agentic GraphRAG.

Usage (from command line)
--------------------------
    python -m agents.pipeline "How do transformers relate to attention mechanisms?"

Usage (from Python code)
--------------------------
    from agents import run_agentic_graphrag

    result = run_agentic_graphrag("What are the key challenges in agentic AI?")
    print(result["answer"])

Output format
-------------
{
    "answer":   str,                    # Final LLM-generated answer
    "sources":  list[dict],             # Deduplicated source citations
    "trace":    list[dict],             # Step-by-step investigation log
    "subgraph": dict,                   # Knowledge graph data accumulated
    "metadata": {
        "steps_taken":          int,
        "tools_used":           list[str],
        "stopped_reason":       str,
        "total_tokens_approx":  int,
        "execution_time_s":     float,
        "evidence_count":       int,
        "entity_count":         int,
    }
}
"""

from __future__ import annotations

import sys
import time
from typing import Any

# Auto-load .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; rely on environment variables being set manually

from agents.orchestrator import AgentOrchestrator
from agents.state import AgentState


def run_agentic_graphrag(
    question: str,
    max_steps: int = 8,
    token_budget: int = 4000,
    evidence_threshold: int = 7,
    backend: str = "gemini",
    model: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run the Agentic GraphRAG pipeline for a question.

    The agent dynamically selects retrieval tools (vector search, entity
    search, graph traversal, document fetch, evidence evaluation) to build
    up evidence, then generates a comprehensive answer.

    Parameters
    ----------
    question : str
        The question to answer.
    max_steps : int
        Maximum tool calls before forcing a stop. Default 8.
    token_budget : int
        Approximate token limit for the whole investigation. Default 4000.
    evidence_threshold : int
        Evidence quality score (0-10) considered "sufficient". Default 7.
    backend : str
        LLM backend: "gemini", "groq", "openai", "anthropic", "mock".
    model : str | None
        Model name override (e.g. "gemini-3.6-flash").
    verbose : bool
        If True, print step-by-step progress.

    Returns
    -------
    dict
        Contains: answer, sources, trace, subgraph, metadata.
    """
    start_time = time.perf_counter()

    orchestrator = AgentOrchestrator(
        max_steps=max_steps,
        token_budget=token_budget,
        evidence_threshold=evidence_threshold,
        backend=backend,
        model=model,
        verbose=verbose,
    )

    state: AgentState = orchestrator.run(question)
    elapsed = time.perf_counter() - start_time

    # Build the tools_used list from the trace
    tools_used = [step.tool_name for step in state.trace]

    # Serialize trace for JSON output
    trace_output = [
        {
            "step": r.step_number,
            "tool": r.tool_name,
            "reasoning": r.reasoning,
            "inputs": r.inputs,
            "tokens_approx": r.tokens_used,
            "output_summary": _summarize(r.outputs),
        }
        for r in state.trace
    ]

    return {
        "answer": state.final_answer,
        "sources": state.final_sources,
        "trace": trace_output,
        "subgraph": {
            "entities": state.subgraph_data.get("entities", []),
            "relationships": state.subgraph_data.get("relationships", []),
            "chunk_ids": state.subgraph_data.get("chunk_ids", []),
        },
        "metadata": {
            "steps_taken": state.step_count,
            "tools_used": tools_used,
            "stopped_reason": state.stopped_reason,
            "total_tokens_approx": state.total_tokens_approx,
            "execution_time_s": round(elapsed, 2),
            "evidence_count": len(state.evidence),
            "entity_count": len(state.subgraph_data.get("entities", [])),
        },
    }


def _summarize(outputs: Any) -> str:
    """Brief string summary of tool outputs for the trace."""
    if outputs is None:
        return "None"
    if isinstance(outputs, list):
        return f"{len(outputs)} item(s)"
    if isinstance(outputs, dict):
        keys = list(outputs.keys())[:4]
        return f"dict with keys: {keys}"
    if isinstance(outputs, str):
        return outputs[:100]
    return str(outputs)[:100]


# ---------------------------------------------------------------------------
# CLI entry point: python -m agents.pipeline "your question"
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Pipeline 3: Agentic GraphRAG — dynamic tool-selection agent."
    )
    parser.add_argument("question", nargs="+", help="The question to answer")
    parser.add_argument(
        "--backend",
        default="gemini",
        choices=["gemini", "groq", "openai", "anthropic", "mock", "auto"],
        help="LLM backend to use (default: gemini)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=8,
        help="Max tool calls before forced stop (default: 8)",
    )
    parser.add_argument(
        "--threshold", type=int, default=7,
        help="Evidence quality score 0-10 to consider sufficient (default: 7)",
    )
    parser.add_argument(
        "--model", default=None,
        help="LLM model name override (default: gemini-3.5-flash-lite)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress step-by-step trace during investigation",
    )
    args = parser.parse_args()

    question = " ".join(args.question)
    print(f"\nRunning Agentic GraphRAG Pipeline...")
    print(f"Question : {question}")
    print(f"Backend  : {args.backend}")
    print(f"Max steps: {args.max_steps}")
    print()

    result = run_agentic_graphrag(
        question=question,
        backend=args.backend,
        max_steps=args.max_steps,
        evidence_threshold=args.threshold,
        model=args.model,
        verbose=not args.quiet,
    )


    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "=" * 60)
    print("SOURCES CITED")
    print("=" * 60)
    for src in result["sources"]:
        print(f"  - {src['chunk_id']}  (doc: {src['doc_id']})")

    print("\n" + "=" * 60)
    print("INVESTIGATION METADATA")
    print("=" * 60)
    meta = result["metadata"]
    print(f"  Steps taken    : {meta['steps_taken']}")
    print(f"  Tools used     : {meta['tools_used']}")
    print(f"  Stopped reason : {meta['stopped_reason']}")
    print(f"  Tokens approx  : {meta['total_tokens_approx']}")
    print(f"  Time (s)       : {meta['execution_time_s']}")
    print(f"  Evidence chunks: {meta['evidence_count']}")
    print(f"  Graph entities : {meta['entity_count']}")

    print("\n" + "=" * 60)
    print("AGENT TRACE (step-by-step investigation)")
    print("=" * 60)
    for step in result["trace"]:
        print(f"\nStep {step['step']}: [{step['tool']}]")
        print(f"  Reasoning : {step['reasoning']}")
        print(f"  Inputs    : {step['inputs']}")
        print(f"  Output    : {step['output_summary']}")
        print(f"  Tokens    : ~{step['tokens_approx']}")
