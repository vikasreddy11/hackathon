import json
from collections import defaultdict

GRAPH_FILE = "evaluation/results/graphrag_results.json"
QUESTION_FILE = "evaluation/questions/eval_public.jsonl"

with open(GRAPH_FILE, "r", encoding="utf-8") as f:
    graph_data = json.load(f)

questions = {}

with open(QUESTION_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            q = json.loads(line)
            questions[q["qid"]] = q

stats = defaultdict(lambda: {
    "total": 0,
    "correct": 0
})

for result in graph_data["questions"]:

    qid = result["qid"]
    qtype = questions[qid]["qtype"]

    stats[qtype]["total"] += 1

    if result["graph_hit"]:
        stats[qtype]["correct"] += 1


print("=" * 60)
print("GRAPHRAG PERFORMANCE BY QUESTION TYPE")
print("=" * 60)

for qtype, values in stats.items():

    total = values["total"]
    correct = values["correct"]

    accuracy = correct / total * 100

    print(
        f"{qtype:15} "
        f"{correct:2}/{total:2} "
        f"({accuracy:.2f}%)"
    )

print("=" * 60)