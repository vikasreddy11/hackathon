"""
evaluation/benchmark.py
=======================
Automated three-way benchmark comparing:
1. Pipeline 1: Standard RAG (FAISS vector search)
2. Pipeline 2: GraphRAG (Knowledge Graph entity + neighborhood retrieval)
3. Pipeline 3: Agentic GraphRAG (Autonomous ReAct multi-step orchestrator)

Usage:
    python -m evaluation.benchmark [--limit 10] [--backend gemini]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from agents.pipeline import run_agentic_graphrag
from evaluation.metrics import exact_or_substring_match, llm_as_judge_eval
from graphrag.pipeline import run_graphrag
from rag.pipeline import run_rag


def load_questions(questions_path: Path) -> list[dict[str, Any]]:
    """Load evaluation questions from jsonl file."""
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    questions = []
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return questions


def run_benchmark(
    questions_path: Path = Path("data/raw/questions/eval_public.jsonl"),
    output_dir: Path = Path("evaluation/results"),
    limit: int | None = None,
    backend: str = "gemini",
    judge_backend: str = "gemini",
) -> dict[str, Any]:
    """Run benchmark across all three pipelines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = load_questions(questions_path)
    if limit:
        questions = questions[:limit]

    print(f"============================================================")
    print(f"Agentic GraphRAG Hackathon — Benchmark Runner")
    print(f"Total questions to evaluate: {len(questions)}")
    print(f"Backend: {backend} | Judge: {judge_backend}")
    print(f"============================================================\n")

    results_detail: list[dict[str, Any]] = []
    stats = {
        "rag": {"correct": 0, "total_time": 0.0, "total_tokens": 0, "scores": []},
        "graphrag": {"correct": 0, "total_time": 0.0, "total_tokens": 0, "scores": []},
        "agentic": {"correct": 0, "total_time": 0.0, "total_tokens": 0, "steps": [], "scores": []},
    }

    for idx, q_item in enumerate(questions, 1):
        qid = q_item.get("qid", f"pub-{idx:03d}")
        question = q_item.get("question", "")
        gold_answers = q_item.get("answer", [])
        qtype = q_item.get("qtype", "general")

        print(f"\n--- [{idx}/{len(questions)}] ({qid}) [{qtype}] {question[:75]}... ---")

        q_result: dict[str, Any] = {
            "qid": qid,
            "question": question,
            "qtype": qtype,
            "gold_answers": gold_answers,
            "pipelines": {},
        }

        # ----------------------------------------------------
        # 1. Standard RAG
        # ----------------------------------------------------
        t0 = time.perf_counter()
        try:
            rag_out = run_rag(question=question, top_k=3, backend=backend)
            rag_time = round(time.perf_counter() - t0, 3)
            rag_answer = rag_out.get("answer", "")
            rag_tokens = len(rag_answer.split()) * 2 + 350
            rag_eval = llm_as_judge_eval(question, rag_answer, gold_answers, backend=judge_backend)
        except Exception as e:
            rag_time = round(time.perf_counter() - t0, 3)
            rag_answer = f"ERROR: {e}"
            rag_tokens = 0
            rag_eval = {"is_correct": False, "score": 1, "reason": str(e)}

        stats["rag"]["total_time"] += rag_time
        stats["rag"]["total_tokens"] += rag_tokens
        stats["rag"]["scores"].append(rag_eval["score"])
        if rag_eval["is_correct"]:
            stats["rag"]["correct"] += 1

        q_result["pipelines"]["rag"] = {
            "answer": rag_answer,
            "execution_time_s": rag_time,
            "tokens": rag_tokens,
            "evaluation": rag_eval,
        }
        print(f"  [RAG]          Score: {rag_eval['score']}/5 | Time: {rag_time}s | Correct: {rag_eval['is_correct']}")

        # ----------------------------------------------------
        # 2. GraphRAG
        # ----------------------------------------------------
        t0 = time.perf_counter()
        try:
            graph_out = run_graphrag(question=question, max_chunks=3, backend=backend)
            graph_time = round(time.perf_counter() - t0, 3)
            graph_answer = graph_out.get("answer", "")
            graph_tokens = len(graph_answer.split()) * 2 + 450
            graph_eval = llm_as_judge_eval(question, graph_answer, gold_answers, backend=judge_backend)
        except Exception as e:
            graph_time = round(time.perf_counter() - t0, 3)
            graph_answer = f"ERROR: {e}"
            graph_tokens = 0
            graph_eval = {"is_correct": False, "score": 1, "reason": str(e)}

        stats["graphrag"]["total_time"] += graph_time
        stats["graphrag"]["total_tokens"] += graph_tokens
        stats["graphrag"]["scores"].append(graph_eval["score"])
        if graph_eval["is_correct"]:
            stats["graphrag"]["correct"] += 1

        q_result["pipelines"]["graphrag"] = {
            "answer": graph_answer,
            "execution_time_s": graph_time,
            "tokens": graph_tokens,
            "evaluation": graph_eval,
        }
        print(f"  [GraphRAG]     Score: {graph_eval['score']}/5 | Time: {graph_time}s | Correct: {graph_eval['is_correct']}")

        # ----------------------------------------------------
        # 3. Agentic GraphRAG
        # ----------------------------------------------------
        t0 = time.perf_counter()
        try:
            agentic_out = run_agentic_graphrag(question=question, max_steps=6, backend=backend, verbose=False)
            agentic_time = round(time.perf_counter() - t0, 3)
            agentic_answer = agentic_out.get("answer", "")
            agentic_trace = agentic_out.get("trace", [])
            agentic_metadata = agentic_out.get("metadata", {})
            agentic_tokens = agentic_metadata.get("total_tokens_approx", len(agentic_answer.split()) * 2 + 400 * max(1, len(agentic_trace)))
            agentic_eval = llm_as_judge_eval(question, agentic_answer, gold_answers, backend=judge_backend)
        except Exception as e:
            agentic_time = round(time.perf_counter() - t0, 3)
            agentic_answer = f"ERROR: {e}"
            agentic_trace = []
            agentic_tokens = 0
            agentic_eval = {"is_correct": False, "score": 1, "reason": str(e)}

        stats["agentic"]["total_time"] += agentic_time
        stats["agentic"]["total_tokens"] += agentic_tokens
        stats["agentic"]["steps"].append(len(agentic_trace))
        stats["agentic"]["scores"].append(agentic_eval["score"])
        if agentic_eval["is_correct"]:
            stats["agentic"]["correct"] += 1

        q_result["pipelines"]["agentic"] = {
            "answer": agentic_answer,
            "execution_time_s": agentic_time,
            "tokens": agentic_tokens,
            "steps": len(agentic_trace),
            "trace_summary": [
                {"step": s.get("step"), "action": s.get("action"), "tool": s.get("tool")}
                for s in agentic_trace
            ],
            "evaluation": agentic_eval,
        }
        print(f"  [Agentic]      Score: {agentic_eval['score']}/5 | Steps: {len(agentic_trace)} | Time: {agentic_time}s | Correct: {agentic_eval['is_correct']}")

        results_detail.append(q_result)

    n = max(1, len(questions))
    summary = {
        "total_evaluated": len(questions),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "rag": {
                "accuracy_pct": round((stats["rag"]["correct"] / n) * 100, 2),
                "avg_score": round(sum(stats["rag"]["scores"]) / n, 2),
                "avg_time_s": round(stats["rag"]["total_time"] / n, 2),
                "avg_tokens": round(stats["rag"]["total_tokens"] / n, 1),
            },
            "graphrag": {
                "accuracy_pct": round((stats["graphrag"]["correct"] / n) * 100, 2),
                "avg_score": round(sum(stats["graphrag"]["scores"]) / n, 2),
                "avg_time_s": round(stats["graphrag"]["total_time"] / n, 2),
                "avg_tokens": round(stats["graphrag"]["total_tokens"] / n, 1),
            },
            "agentic": {
                "accuracy_pct": round((stats["agentic"]["correct"] / n) * 100, 2),
                "avg_score": round(sum(stats["agentic"]["scores"]) / n, 2),
                "avg_time_s": round(stats["agentic"]["total_time"] / n, 2),
                "avg_tokens": round(stats["agentic"]["total_tokens"] / n, 1),
                "avg_steps": round(sum(stats["agentic"]["steps"]) / n, 2) if stats["agentic"]["steps"] else 0,
            },
        },
    }

    # Save detailed and summary JSON files
    detail_path = output_dir / "benchmark_100_results.json"
    summary_path = output_dir / "benchmark_summary.json"

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results_detail, f, indent=2, ensure_ascii=False)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n============================================================")
    print("BENCHMARK RESULTS SUMMARY")
    print("============================================================")
    for p_name, p_metrics in summary["metrics"].items():
        print(f"[{p_name.upper()}] Accuracy: {p_metrics['accuracy_pct']}% | Avg Score: {p_metrics['avg_score']}/5 | Avg Time: {p_metrics['avg_time_s']}s | Avg Tokens: {p_metrics['avg_tokens']}")
    print(f"\nDetailed results saved to: {detail_path}")
    print(f"Summary saved to:          {summary_path}")
    print("============================================================\n")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 3-pipeline benchmark.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/raw/questions/eval_public.jsonl"),
        help="Path to eval_public.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory to save results",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions to evaluate (for quick test)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gemini",
        help="LLM backend for pipelines (gemini, groq, openai, mock)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        questions_path=args.questions,
        output_dir=args.output_dir,
        limit=args.limit,
        backend=args.backend,
    )
