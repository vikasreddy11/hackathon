import json
from collections import Counter

FILE = "evaluation/questions/eval_public.jsonl"

questions = []

with open(FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            questions.append(json.loads(line))

print("=" * 50)
print("DATASET ANALYSIS")
print("=" * 50)

print(f"Total questions: {len(questions)}")

qtypes = Counter(q["qtype"] for q in questions)

print("\nQuestion types:")
for qtype, count in qtypes.items():
    print(f"  {qtype}: {count}")

print("\nExample questions:")

for q in questions[:5]:
    print(f"\n{q['qid']}")
    print(f"Type: {q['qtype']}")
    print(f"Question: {q['question']}")
    print(f"Answer: {q['answer']}")

print("\n" + "=" * 50)