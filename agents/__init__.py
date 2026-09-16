"""
agents/__init__.py
==================
Agentic GraphRAG — Pipeline 3.

The orchestrator agent dynamically selects retrieval tools based on the
question and accumulated evidence, instead of following a fixed sequence.

Public API
----------
    from agents import run_agentic_graphrag

    result = run_agentic_graphrag("How do transformers relate to attention?")
    print(result["answer"])
    for step in result["trace"]:
        print(step)
"""

from agents.pipeline import run_agentic_graphrag
from agents.orchestrator import AgentOrchestrator
from agents.state import AgentState, ToolResult

__all__ = [
    "run_agentic_graphrag",
    "AgentOrchestrator",
    "AgentState",
    "ToolResult",
]
