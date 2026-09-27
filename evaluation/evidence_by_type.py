import json
from collections import defaultdict

RAG_FILE = "evaluation/results/retrieval_results.json"
GRAPH_FILE = "evaluation/results/graphrag_results.json"
QUESTIONS_FILE = "evaluation/questions/eval_public.jsonl"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


rag = load_json(RAG_FILE)
graph = load_json(GRAPH_FILE)

questions = {}

with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            q = json.loads(line)
            questions[q["qid"]] = q


stats = defaultdict(
    lambda: {
        "total": 0,
        "rag_supported": 0,
        "graph_supported": 0
    }
)


for qid, question in questions.items():

    qtype = question["qtype"]

    rag_result = next(
        r for r in rag["questions"]
        if r["qid"] == qid
    )

    graph_result = next(
        r for r in graph["questions"]
        if r["qid"] == qid
    )

    stats[qtype]["total"] += 1

    if rag_result["retrieval_hit"]:
        stats[qtype]["rag_supported"] += 1

    if graph_result["graph_hit"]:
        stats[qtype]["graph_supported"] += 1


print("=" * 70)
print("EVIDENCE COVERAGE BY QUESTION TYPE")
print("=" * 70)

print(
    f"{'Question Type':<18}"
    f"{'Total':<8}"
    f"{'RAG':<12}"
    f"{'GraphRAG':<12}"
)

print("-" * 70)


for qtype, values in stats.items():

    total = values["total"]

    rag_percentage = (
        values["rag_supported"] / total * 100
    )

    graph_percentage = (
        values["graph_supported"] / total * 100
    )

    print(
        f"{qtype:<18}"
        f"{total:<8}"
        f"{rag_percentage:.2f}%"
        f"{graph_percentage:.2f}%"
    )


print("=" * 70)