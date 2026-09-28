import json
import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

from graph.graph_search import get_graph_answer


QUESTION_FILE = "evaluation/questions/eval_public.jsonl"
OUTPUT_FILE = "evaluation/results/graphrag_results.json"


def normalize(text):
    return str(text).strip().lower()


def check_answer(predicted, expected):
    if predicted is None:
        return False

    predicted = normalize(predicted)

    if isinstance(expected, list):
        answers = expected
    else:
        answers = [expected]

    for answer in answers:
        if normalize(answer) in predicted:
            return True

    return False


# ============================================================
# LOAD QUESTIONS
# ============================================================

with open(QUESTION_FILE, "r", encoding="utf-8") as f:
    questions = [
        json.loads(line)
        for line in f
        if line.strip()
    ]


# ============================================================
# EVALUATION
# ============================================================

total = len(questions)
correct = 0
results = []

print("=" * 70)
print("GRAPHRAG EVALUATION")
print("=" * 70)


for index, question in enumerate(questions, start=1):

    qid = question["qid"]
    text = question["question"]
    expected = question["answer"]

    print(f"\n[{index}/{total}] {qid}")
    print(f"Question: {text}")

    graph_answer = get_graph_answer(text)

    # --------------------------------------------------------
    # GET PREDICTED ANSWER
    # --------------------------------------------------------

    if graph_answer:

        predicted = graph_answer.get("answer")

        # Fallback for older graph results
        if predicted is None:
            predicted = graph_answer.get("gold", "")

    else:
        predicted = ""

    # --------------------------------------------------------
    # CHECK ANSWER
    # --------------------------------------------------------

    is_correct = check_answer(
        predicted,
        expected
    )

    if is_correct:
        correct += 1

    print(f"GraphRAG Answer: {predicted}")
    print(
        f"Correct: {'YES' if is_correct else 'NO'}"
    )

    results.append({
        "qid": qid,
        "question": text,
        "qtype": question["qtype"],
        "graphrag_answer": predicted,
        "gold_answer": expected,
        "correct": is_correct,
        "graph_result": graph_answer
    })


# ============================================================
# SUMMARY
# ============================================================

accuracy = (
    correct / total * 100
    if total > 0
    else 0
)

summary = {
    "total": total,
    "correct": correct,
    "accuracy": accuracy
}


# ============================================================
# SAVE RESULTS
# ============================================================

output = {
    "summary": summary,
    "questions": results
}

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


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("FINAL GRAPHRAG RESULTS")
print("=" * 70)

print(f"Total questions : {total}")
print(f"Correct         : {correct}")
print(f"Accuracy        : {accuracy:.2f}%")

print("\nResults saved to:")
print(OUTPUT_FILE)

print("=" * 70)