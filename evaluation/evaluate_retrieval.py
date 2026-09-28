import json
import sys
import os

# Allow Python to find the project modules
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from rag.search import search


FILE = "evaluation/questions/eval_public.jsonl"


with open(FILE, "r", encoding="utf-8") as f:
    questions = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


total = len(questions)
correct = 0

results = []


print("=" * 70)
print("FAST RAG RETRIEVAL EVALUATION")
print("=" * 70)

for index, q in enumerate(questions, start=1):

    print(f"[{index}/{total}] {q['qid']}")

    retrieved = search(
        q["question"],
        top_k=5
    )

    retrieved_doc_ids = [
        result["document"]["doc_id"]
        for result in retrieved
    ]

    gold_doc_ids = q["gold_doc_ids"]

    hit = any(
        doc_id in gold_doc_ids
        for doc_id in retrieved_doc_ids
    )

    if hit:
        correct += 1

    results.append({
        "qid": q["qid"],
        "question": q["question"],
        "gold_doc_ids": gold_doc_ids,
        "retrieved_doc_ids": retrieved_doc_ids,
        "retrieval_hit": hit
    })


accuracy = correct / total * 100


print("\n" + "=" * 70)
print("FINAL RETRIEVAL RESULTS")
print("=" * 70)

print(f"Total questions : {total}")
print(f"Correct retrieval: {correct}")
print(f"Retrieval Recall@5: {accuracy:.2f}%")
print("=" * 70)


OUTPUT_FILE = "evaluation/results/retrieval_results.json"

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        {
            "summary": {
                "total": total,
                "correct": correct,
                "recall_at_5": accuracy
            },
            "questions": results
        },
        f,
        indent=2,
        ensure_ascii=False
    )


print(f"\nResults saved to: {OUTPUT_FILE}")