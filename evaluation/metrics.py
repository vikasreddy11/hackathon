"""
evaluation/metrics.py
=====================
Scoring metrics for comparing RAG, GraphRAG, and Agentic GraphRAG pipelines:
- Substring / Exact match against ground truth
- LLM-as-a-Judge accuracy score (1-5 scale and PASS/FAIL)
- Token efficiency calculations
"""

from __future__ import annotations

import json
import re
from typing import Any

from rag.llm_client import call_llm


def exact_or_substring_match(prediction: str, gold_answers: list[str]) -> bool:
    """Check if any gold answer is directly contained in the prediction text."""
    if not prediction or not gold_answers:
        return False
    pred_lower = prediction.lower().strip()
    for gold in gold_answers:
        gold_clean = str(gold).lower().strip()
        if gold_clean and gold_clean in pred_lower:
            return True
    return False


def llm_as_judge_eval(
    question: str,
    prediction: str,
    gold_answers: list[str],
    backend: str = "gemini",
) -> dict[str, Any]:
    """
    Use an LLM-as-a-judge to evaluate if the predicted answer accurately
    answers the question and matches the ground truth.
    """
    gold_str = " OR ".join(str(g) for g in gold_answers)
    prompt = f"""You are an objective AI evaluation judge. Compare the model's generated answer against the ground truth answer.

Question: {question}
Ground Truth: {gold_str}
Model Generated Answer: {prediction}

Evaluate the generated answer on:
1. Correctness: Does it state the correct factual answer matching the ground truth? (Pass/Fail)
2. Completeness: Does it answer the question completely?
3. Score: Rate from 1 to 5 (5 = completely correct and clear, 1 = completely incorrect or hallucinated).

Respond strictly in this JSON format:
{{
  "is_correct": true or false,
  "score": <int between 1 and 5>,
  "reason": "<one-sentence rationale>"
}}
"""
    try:
        raw_res = call_llm(prompt, backend=backend, max_tokens=150, temperature=0.0)
        json_match = re.search(r"\{.*\}", raw_res, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        match = exact_or_substring_match(prediction, gold_answers)
        return {
            "is_correct": match,
            "score": 5 if match else 1,
            "reason": f"Heuristic match fallback (error: {e})",
        }

    match = exact_or_substring_match(prediction, gold_answers)
    return {
        "is_correct": match,
        "score": 5 if match else 1,
        "reason": "Deterministic fallback match",
    }
