import json
from collections import defaultdict

FILE = "evaluation/results/graphrag_results.json"

with open(FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

questions = data["questions"]

stats = defaultdict(lambda: {"correct": 0, "total": 0})

for result in questions:
    qtype = result["qtype"]

    stats[qtype]["total"] += 1

    if result["correct"]:
        stats[qtype]["correct"] += 1


print("=" * 70)
print("GRAPHRAG CATEGORY ANALYSIS")
print("=" * 70)

for qtype, values in stats.items():

    total = values["total"]
    correct = values["correct"]

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    print(
        f"{qtype:<15} "
        f"{correct}/{total} "
        f"({accuracy:.2f}%)"
    )

print("=" * 70)