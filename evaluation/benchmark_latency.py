import json
import sys
import os
import time
import statistics

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from rag.search import search
from graph.graph_search import graph_search


FILE = "evaluation/questions/eval_public.jsonl"


with open(FILE, "r", encoding="utf-8") as f:
    questions = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


rag_times = []
graph_times = []

print("=" * 70)
print("LATENCY BENCHMARK")
print("=" * 70)

for index, q in enumerate(questions, start=1):

    question = q["question"]

    # RAG latency
    start = time.perf_counter()

    search(question, top_k=5)

    rag_time = time.perf_counter() - start
    rag_times.append(rag_time)

    # GraphRAG latency
    start = time.perf_counter()

    graph_search(question)

    graph_time = time.perf_counter() - start
    graph_times.append(graph_time)

    print(
        f"[{index}/{len(questions)}] "
        f"RAG: {rag_time:.4f}s | "
        f"GraphRAG: {graph_time:.4f}s"
    )


rag_avg = statistics.mean(rag_times)
graph_avg = statistics.mean(graph_times)


print("\n" + "=" * 70)
print("FINAL LATENCY RESULTS")
print("=" * 70)

print(f"Questions tested       : {len(questions)}")
print(f"Average RAG latency    : {rag_avg:.4f} seconds")
print(f"Average GraphRAG latency: {graph_avg:.4f} seconds")

print("=" * 70)


output = {
    "questions_tested": len(questions),
    "rag": {
        "average_latency_seconds": rag_avg
    },
    "graphrag": {
        "average_latency_seconds": graph_avg
    }
}


OUTPUT_FILE = "evaluation/results/latency_results.json"

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        indent=2
    )

print(
    f"\nResults saved to: {OUTPUT_FILE}"
)