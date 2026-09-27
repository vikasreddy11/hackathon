import json


RAG_FILE = "evaluation/results/retrieval_results.json"
GRAPH_FILE = "evaluation/results/graphrag_results.json"


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


rag = load_json(RAG_FILE)
graph = load_json(GRAPH_FILE)


def calculate_quality(data, field):
    total = len(data["questions"])
    supported = sum(
        1
        for result in data["questions"]
        if result[field]
    )

    return supported / total * 100


rag_quality = calculate_quality(
    rag,
    "retrieval_hit"
)

graph_quality = calculate_quality(
    graph,
    "graph_hit"
)


print("=" * 60)
print("EVIDENCE QUALITY EVALUATION")
print("=" * 60)

print(f"RAG evidence coverage      : {rag_quality:.2f}%")
print(f"GraphRAG evidence coverage : {graph_quality:.2f}%")

print("=" * 60)