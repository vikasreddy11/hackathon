"""
rag/llm_client.py
=================
Multi-backend LLM client supporting OpenAI, Groq, Google Gemini, Anthropic,
and a deterministic extractive-synthesis mock backend for offline/test usage.
"""

from __future__ import annotations

import os
import re
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _extractive_mock_answer(question: str, prompt: str) -> str:
    """
    Generate a coherent, deterministic synthesis based on prompt context
    when no remote LLM API keys are configured.
    """
    clean_prompt = prompt.strip()
    # Extract lines or sentences from context chunks
    context_lines = []
    for line in clean_prompt.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Skip standard prompt headers
        if any(line.lower().startswith(h) for h in [
            "question:", "system:", "context:", "instruction:", "user:",
            "you are", "answer the", "based on", "subgraph:", "entities:"
        ]):
            continue
        if len(line) > 30 and not line.startswith("---") and not line.startswith("==="):
            context_lines.append(line)

    if not context_lines:
        return f"Based on the provided information, regarding '{question}': relevant details were retrieved."

    # Pick the most relevant 2-4 lines containing terms from the question
    q_words = set(re.findall(r"\w+", question.lower()))
    scored_lines = []
    for cl in context_lines:
        cl_words = set(re.findall(r"\w+", cl.lower()))
        overlap = len(q_words & cl_words)
        scored_lines.append((overlap, cl))

    scored_lines.sort(key=lambda x: x[0], reverse=True)
    selected = [line for score, line in scored_lines[:3]]
    if not selected:
        selected = context_lines[:3]

    summary = " ".join(selected)
    return (
        f"Based on the provided context regarding '{question}', here is the synthesized answer:\n\n"
        f"{summary}\n\n"
        f"This conclusion is grounded in the retrieved sources."
    )


class LLMClient:
    """
    Unified LLM caller with pluggable backends.
    Supported backends: 'auto', 'mock', 'groq', 'openai', 'gemini', 'anthropic'.
    """

    def __init__(self, default_backend: str = "auto", default_model: str | None = None):
        self.default_backend = default_backend.lower()
        self.default_model = default_model

    def resolve_backend(self, backend: str | None = None) -> str:
        target = (backend or self.default_backend or os.getenv("LLM_BACKEND", "auto")).lower()
        if target != "auto":
            return target

        # Auto-detect available API keys
        if os.getenv("GROQ_API_KEY"):
            return "groq"
        if os.getenv("OPENAI_API_KEY"):
            return "openai"
        if os.getenv("GEMINI_API_KEY"):
            return "gemini"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "mock"

    def generate(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful, factual assistant that answers questions using the provided context.",
        backend: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        resolved = self.resolve_backend(backend)
        target_model = model or self.default_model

        try:
            if resolved == "groq":
                return self._call_groq(prompt, system_prompt, target_model or "llama-3.1-8b-instant", temperature, max_tokens)
            elif resolved == "openai":
                return self._call_openai(prompt, system_prompt, target_model or "gpt-4o-mini", temperature, max_tokens)
            elif resolved == "gemini":
                default_gemini = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
                return self._call_gemini(prompt, system_prompt, target_model or default_gemini, temperature, max_tokens)
            elif resolved == "anthropic":
                return self._call_anthropic(prompt, system_prompt, target_model or "claude-3-5-haiku-20241022", temperature, max_tokens)
            else:
                # Default / mock
                return _extractive_mock_answer(prompt, prompt)
        except Exception as exc:
            # Graceful fallback to extractive mock with warning in case of network/key errors
            print(f"[LLMClient WARNING] Failed calling backend '{resolved}': {exc}. Falling back to mock synthesizer.")
            return _extractive_mock_answer(prompt, prompt)

    def _call_groq(self, prompt: str, system_prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        from groq import Groq
        client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content or ""

    def _call_openai(self, prompt: str, system_prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return completion.choices[0].message.content or ""

    def _call_gemini(self, prompt: str, system_prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        import time
        from google import genai
        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        full_content = f"{system_prompt}\n\n{prompt}"
        for attempt in range(4):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=full_content,
                )
                return response.text or ""
            except Exception as exc:
                if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                    if attempt < 3:
                        wait_sec = 20
                        print(f"\n[Gemini Free Tier] Rate limit reached. Pausing {wait_sec}s before retry ({attempt + 1}/3)...")
                        time.sleep(wait_sec)
                        continue
                raise

    def _call_anthropic(self, prompt: str, system_prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text


# Global default instance
_default_client = LLMClient()


def call_llm(
    prompt: str,
    system_prompt: str = "You are a helpful, factual assistant that answers questions using the provided context.",
    backend: str | None = None,
    model: str | None = None,
    **kwargs: Any,
) -> str:
    """Convenience helper to generate LLM responses."""
    return _default_client.generate(
        prompt=prompt,
        system_prompt=system_prompt,
        backend=backend,
        model=model,
        **kwargs,
    )
