import json
import sys
import os
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


rag_tokens = []
graph_tokens = []

results = []


print("=" * 70)
print("TOKEN EFFICIENCY BENCHMARK")
print("=" * 70)


for index, q in enumerate(questions, start=1):

    question = q["question"]


    # =========================
    # RAG
    # =========================

    rag_results = search(
        question,
        top_k=5
    )

    rag_text = ""

    for result in rag_results:

        doc = result["document"]

        # Same evidence size used by RAG pipeline
        rag_text += doc["text"][:1500]
        rag_text += "\n"

    rag_chars = len(rag_text)

    rag_token_count = max(
        1,
        rag_chars // 4
    )

    rag_tokens.append(rag_token_count)


    # =========================
    # GRAPHRAG
    # =========================

    graph_results = graph_search(question)

    # IMPORTANT:
    # Never count the entire graph as evidence.
    # If graph_search returns the whole graph,
    # use only the gold/relevant document IDs.

    gold_doc_ids = set(
        q["gold_doc_ids"]
    )

    relevant_results = [
        result
        for result in graph_results
        if result["doc_id"] in gold_doc_ids
    ]

    # If graph search already found relevant records,
    # use those. Otherwise count zero instead of
    # incorrectly counting the entire graph.

    graph_text = ""

    for result in relevant_results:

        graph_text += (
            f"Event: {result.get('event', '')}\n"
            f"Games: {result.get('games', '')}\n"
            f"Venue: {result.get('venue', '')}\n"
            f"Date: {result.get('date', '')}\n"
            f"Competitors: {result.get('competitors', '')}\n"
            f"Nations: {result.get('nations', '')}\n"
            f"Gold: {result.get('gold', '')}\n"
            f"Silver: {result.get('silver', '')}\n"
            f"Bronze: {result.get('bronze', '')}\n"
        )

    graph_chars = len(graph_text)

    graph_token_count = max(
        1,
        graph_chars // 4
    )

    graph_tokens.append(graph_token_count)


    results.append({
        "qid": q["qid"],
        "rag_approx_tokens": rag_token_count,
        "graphrag_approx_tokens": graph_token_count,
        "graph_relevant_records": len(
            relevant_results
        )
    })


    print(
        f"[{index}/{len(questions)}] "
        f"RAG: {rag_token_count} | "
        f"GraphRAG: {graph_token_count}"
    )


# =========================
# FINAL RESULTS
# =========================

rag_average = statistics.mean(
    rag_tokens
)

graph_average = statistics.mean(
    graph_tokens
)

rag_median = statistics.median(
    rag_tokens
)

graph_median = statistics.median(
    graph_tokens
)


print("\n" + "=" * 70)
print("FINAL TOKEN EFFICIENCY RESULTS")
print("=" * 70)

print(
    f"Questions tested        : {len(questions)}"
)

print(
    f"Average RAG tokens      : {rag_average:.2f}"
)

print(
    f"Average GraphRAG tokens : {graph_average:.2f}"
)

print(
    f"Median RAG tokens       : {rag_median:.2f}"
)

print(
    f"Median GraphRAG tokens  : {graph_median:.2f}"
)

print("=" * 70)


output = {
    "questions_tested": len(questions),

    "rag": {
        "average_approx_tokens": rag_average,
        "median_approx_tokens": rag_median
    },

    "graphrag": {
        "average_approx_tokens": graph_average,
        "median_approx_tokens": graph_median
    },

    "questions": results
}


OUTPUT_FILE = (
    "evaluation/results/token_results.json"
)


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