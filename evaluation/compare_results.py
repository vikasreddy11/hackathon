import json

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


stats = {}

for qtype in set(
    questions[qid]["qtype"]
    for qid in questions
):

    rag_correct = 0
    graph_correct = 0
    total = 0

    for qid, q in questions.items():

        if q["qtype"] != qtype:
            continue

        total += 1

        rag_result = next(
            x for x in rag["questions"]
            if x["qid"] == qid
        )

        graph_result = next(
            x for x in graph["questions"]
            if x["qid"] == qid
        )

        if rag_result["retrieval_hit"]:
            rag_correct += 1

        if graph_result["graph_hit"]:
            graph_correct += 1

    stats[qtype] = {
        "total": total,
        "rag": rag_correct / total * 100,
        "graphrag": graph_correct / total * 100
    }


print("=" * 70)
print("RAG vs GRAPHRAG COMPARISON")
print("=" * 70)

print(
    f"{'Question Type':<18}"
    f"{'Total':<8}"
    f"{'RAG':<12}"
    f"{'GraphRAG':<12}"
)

print("-" * 70)

for qtype, values in stats.items():

    print(
        f"{qtype:<18}"
        f"{values['total']:<8}"
        f"{values['rag']:.2f}%"
        f"{values['graphrag']:.2f}%"
    )

print("-" * 70)

print(
    f"{'Overall':<18}"
    f"{100:<8}"
    f"{rag['summary']['recall_at_5']:.2f}%"
    f"{graph['summary']['recall_at_5']:.2f}%"
)

print("=" * 70)