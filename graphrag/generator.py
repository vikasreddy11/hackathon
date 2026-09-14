"""
graphrag/generator.py
=====================
Constructs graph-augmented prompts and generates answers using the LLM client.
"""

from __future__ import annotations

from typing import Any

from rag.llm_client import call_llm


def build_graphrag_prompt(question: str, graph_context: str) -> str:
    """Construct prompt combining question with graph entities, relations, and evidence."""
    return (
        f"You are an expert analytical assistant with access to a domain Knowledge Graph.\n\n"
        f"--- GRAPH CONTEXT START ---\n"
        f"{graph_context}\n"
        f"--- GRAPH CONTEXT END ---\n\n"
        f"Question: {question.strip()}\n\n"
        f"Instructions:\n"
        f"1. Synthesize the answer using the identified entities, relational connections, and evidence.\n"
        f"2. Explicitly explain relevant relationships (e.g. how entities connect or co-occur).\n"
        f"3. Cite supporting chunks when detailing factual claims.\n"
        f"4. If the graph context lacks sufficient information, state that clearly."
    )


def generate_graphrag_answer(
    question: str,
    graph_context: str,
    backend: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate an answer given a question and structured graph context.

    Parameters
    ----------
    question : str
        The query to answer.
    graph_context : str
        Formatted graph context from build_graph_context().
    backend : str | None
        LLM backend ('groq', 'openai', 'gemini', 'anthropic', 'mock', 'auto').
    model : str | None
        Model identifier override.
    system_prompt : str | None
        System prompt override.

    Returns
    -------
    str
        Generated answer string.
    """
    prompt = build_graphrag_prompt(question, graph_context)
    sys_prompt = system_prompt or (
        "You are an expert knowledge graph reasoning assistant. Use graph relations "
        "and supporting evidence to provide comprehensive, factual explanations."
    )
    return call_llm(
        prompt=prompt,
        system_prompt=sys_prompt,
        backend=backend,
        model=model,
        **kwargs,
    )
