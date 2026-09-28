"""
agents/evaluator.py
===================
EvidenceEvaluator — asks Gemini to judge whether accumulated evidence is
sufficient to confidently answer the original question.

This is separate from the evaluate_evidence TOOL (which is callable by the
orchestrator as one of its 5 options). This class is the standalone evaluator
used by the orchestrator's stopping-criteria check.

Both use the same underlying logic — this just provides a clean class API.
"""

from __future__ import annotations

from typing import Any

from agents.tools import evaluate_evidence


class EvidenceEvaluator:
    """
    Wraps the evaluate_evidence tool in a class for convenient reuse.

    Usage
    -----
        evaluator = EvidenceEvaluator(threshold=7)
        result = evaluator.evaluate(question, evidence_list)
        if result["ready"]:
            # generate final answer
        else:
            print("Still missing:", result["missing"])
    """

    def __init__(
        self,
        threshold: int = 7,
        backend: str | None = "gemini",
        model: str | None = None,
    ):
        """
        Parameters
        ----------
        threshold : int
            Minimum score (0-10) to consider evidence "ready". Default is 7.
        backend : str | None
            LLM backend for scoring (default: "gemini").
        model : str | None
            Model override.
        """
        self.threshold = threshold
        self.backend = backend
        self.model = model

    def evaluate(
        self,
        question: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Evaluate the evidence against the question.

        Parameters
        ----------
        question : str
            The original user question.
        evidence : list[dict]
            Accumulated chunks from AgentState.evidence.

        Returns
        -------
        dict with keys:
            score   : int (0-10)
            ready   : bool (True if score >= threshold)
            missing : list[str] (what is still missing)
            reasoning : str
        """
        tool_result = evaluate_evidence(
            question=question,
            evidence=evidence,
            backend=self.backend,
            model=self.model,
            reasoning="Automatic evidence quality check by EvidenceEvaluator",
        )
        result: dict[str, Any] = tool_result.outputs  # type: ignore[assignment]

        # Apply the configured threshold (may differ from tool's default of 7)
        result["ready"] = result.get("score", 0) >= self.threshold
        return result
