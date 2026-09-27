import json
import sys
import os

# Allow Python to find files inside the rag folder
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from rag.rag_pipeline import rag_pipeline


FILE = "evaluation/questions/eval_public.jsonl"


def normalize(text):
    return str(text).strip().lower()


def check_answer(predicted, expected):
    predicted = normalize(predicted)

    return any(
        normalize(answer) in predicted
        for answer in expected
    )


def rag_answer(question):
    answer, context = rag_pipeline(question)
    return answer, context


with open(FILE, "r", encoding="utf-8") as f:
    questions = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


results = {
    "rag": {
        "correct": 0,
        "total": len(questions)
    }
}

all_results = []


print("=" * 70)
print("RAG EVALUATION")
print("=" * 70)

for index, q in enumerate(questions, start=1):

    print(
        f"\n[{index}/{len(questions)}] "
        f"{q['qid']}"
    )

    answer, context = rag_answer(
        q["question"]
    )

    correct = check_answer(
        answer,
        q["answer"]
    )

    if correct:
        results["rag"]["correct"] += 1

    print(
        f"Correct: {'YES' if correct else 'NO'}"
    )

    all_results.append({
        "qid": q["qid"],
        "question": q["question"],
        "rag_answer": answer,
        "gold_answer": q["answer"],
        "correct": correct
    })


total = results["rag"]["total"]
correct = results["rag"]["correct"]

accuracy = (
    correct / total * 100
)


print("\n" + "=" * 70)
print("FINAL RAG RESULTS")
print("=" * 70)

print(f"Total questions : {total}")
print(f"Correct         : {correct}")
print(f"Accuracy        : {accuracy:.2f}%")

output = {
    "summary": {
        "rag": {
            "correct": correct,
            "total": total,
            "accuracy": accuracy
        }
    },
    "questions": all_results
}


OUTPUT_FILE = (
    "evaluation/results/results.json"
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )

print(
    f"\nResults saved to: {OUTPUT_FILE}"
)