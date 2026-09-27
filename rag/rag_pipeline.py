from rag.search import search
from ollama import chat


def build_context(results):
    context_parts = []

    for rank, result in enumerate(results, start=1):
        doc = result["document"]

        context_parts.append(
            f"""
SOURCE {rank}
Title: {doc["title"]}
Doc ID: {doc["doc_id"]}
URL: {doc["url"]}

Evidence:
{doc["text"][:1500]}
"""
        )

    return "\n".join(context_parts)


def rag_pipeline(question):

    # Retrieve evidence
    results = search(question, top_k=3)

    # Build context
    context = build_context(results)

    # Ask Llama using retrieved evidence
    prompt = f"""
Answer the question using ONLY the evidence provided below.

Question:
{question}

Evidence:
{context}

Give a concise answer.
"""

    response = chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response.message.content

    return answer, context


if __name__ == "__main__":

    question = input("\nEnter your question: ")

    answer, context = rag_pipeline(question)

    print("\n" + "=" * 70)
    print("RAG ANSWER")
    print("=" * 70)

    print(answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)

    print(context)