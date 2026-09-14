"""
rag/generator.py
================
Constructs augmented generation prompts and synthesizes answers using
the swappable LLM client.
"""

from __future__ import annotations

from typing import Any

from rag.llm_client import call_llm


def format_context_chunks(chunks: list[dict[str, Any]]) -> str:
    """Format a list of chunk dicts into structured context text for the prompt."""
    if not chunks:
        return "No relevant context found."

    formatted_parts: list[str] = []
    for idx, c in enumerate(chunks, 1):
        chunk_id = c.get("chunk_id", f"chunk_{idx}")
        doc_id = c.get("doc_id", "unknown_doc")
        score = c.get("score")
        score_str = f" (relevance score: {score:.4f})" if score is not None else ""
        text = c.get("text", "").strip()

        formatted_parts.append(
            f"[Source {idx}: {chunk_id} | Doc: {doc_id}{score_str}]\n{text}"
        )

    return "\n\n".join(formatted_parts)


def build_rag_prompt(question: str, context_chunks: list[dict[str, Any]]) -> str:
    """Construct the user prompt containing grounding context and the question."""
    context_text = format_context_chunks(context_chunks)
    prompt = (
        f"You are given the following context from retrieved documents:\n\n"
        f"--- CONTEXT START ---\n"
        f"{context_text}\n"
        f"--- CONTEXT END ---\n\n"
        f"Question: {question.strip()}\n\n"
        f"Instructions:\n"
        f"Answer the question thoroughly and factually using only the provided context.\n"
        f"If the context does not contain sufficient details to answer, state that clearly."
    )
    return prompt


def generate_answer(
    question: str,
    context_chunks: list[dict[str, Any]],
    backend: str | None = None,
    model: str | None = None,
    system_prompt: str | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate an answer given a question and a list of retrieved chunk dicts.

    Parameters
    ----------
    question : str
        The query to answer.
    context_chunks : list[dict[str, Any]]
        List of chunks returned by RAGRetriever.
    backend : str | None
        LLM backend ('groq', 'openai', 'gemini', 'anthropic', 'mock', or 'auto').
    model : str | None
        Specific model name override.
    system_prompt : str | None
        Optional system prompt override.

    Returns
    -------
    str
        Generated answer.
    """
    prompt = build_rag_prompt(question, context_chunks)
    sys_prompt = system_prompt or (
        "You are an expert technical assistant. Answer questions accurately "
        "and concisely using the provided context."
    )
    return call_llm(
        prompt=prompt,
        system_prompt=sys_prompt,
        backend=backend,
        model=model,
        **kwargs,
    )
