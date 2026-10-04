"""
evaluation/run_hidden.py
========================
Runs the 50 held-out / hidden evaluation questions through the Agentic GraphRAG
pipeline and exports the official submission JSON file with generated answers,
token consumption metrics, and step-by-step agentic traces.

Usage:
    python -m evaluation.run_hidden [--backend gemini] [--output submission_hidden_50.json]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from agents.pipeline import run_agentic_graphrag


def load_hidden_questions(path: Path) -> list[dict[str, Any]]:
    """Load hidden questions from jsonl file."""
    if not path.exists():
        raise FileNotFoundError(f"Hidden questions file not found: {path}")
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return questions


def run_hidden_evaluation(
    questions_path: Path = Path("data/raw/questions/eval_hidden.jsonl"),
    output_path: Path = Path("evaluation/results/submission_hidden_50.json"),
    backend: str = "gemini",
    max_steps: int = 8,
) -> list[dict[str, Any]]:
    """Run all 50 hidden questions through Agentic GraphRAG and export results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    questions = load_hidden_questions(questions_path)

    print(f"============================================================")
    print(f"Agentic GraphRAG Hackathon — Hidden Questions Submission Runner")
    print(f"Total hidden questions: {len(questions)}")
    print(f"Backend: {backend}")
    print(f"============================================================\n")

    submission_records: list[dict[str, Any]] = []

    for idx, q_item in enumerate(questions, 1):
        qid = q_item.get("qid", f"hidden-{idx:03d}")
        question = q_item.get("question", "")

        print(f"[{idx:02d}/{len(questions):02d}] Evaluating {qid}: {question[:75]}...")

        t0 = time.perf_counter()
        try:
            res = run_agentic_graphrag(question=question, max_steps=max_steps, backend=backend, verbose=False)
            exec_time = round(time.perf_counter() - t0, 3)
            answer = res.get("answer", "")
            trace = res.get("trace", [])
            metadata = res.get("metadata", {})
            tokens_used = metadata.get("total_tokens_approx", len(answer.split()) * 2 + 400 * max(1, len(trace)))
            stop_reason = metadata.get("stopped_reason", "completed")
        except Exception as e:
            exec_time = round(time.perf_counter() - t0, 3)
            answer = f"ERROR: {e}"
            trace = []
            tokens_used = 0
            stop_reason = f"error: {e}"

        record = {
            "qid": qid,
            "question": question,
            "answer": answer,
            "tokens_used": {
                "estimated_total_tokens": tokens_used,
            },
            "execution_time_s": exec_time,
            "agentic_trace": {
                "num_steps": len(trace),
                "stop_reason": stop_reason,
                "steps": trace,
            },
        }

        submission_records.append(record)
        print(f"       -> Done in {exec_time}s | Steps: {len(trace)} | Tokens: {tokens_used}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(submission_records, f, indent=2, ensure_ascii=False)

    print(f"\n============================================================")
    print(f"Submission file successfully generated:")
    print(f"-> {output_path.resolve()}")
    print(f"Total questions recorded: {len(submission_records)}")
    print(f"============================================================\n")

    return submission_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hidden questions and generate submission file.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/raw/questions/eval_hidden.jsonl"),
        help="Path to eval_hidden.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evaluation/results/submission_hidden_50.json"),
        help="Path to save output JSON",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        help="LLM backend (gemini, groq, openai, mock)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=8,
        help="Maximum ReAct steps for Agentic GraphRAG",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_hidden_evaluation(
        questions_path=args.questions,
        output_path=args.output,
        backend=args.backend,
        max_steps=args.max_steps,
    )
