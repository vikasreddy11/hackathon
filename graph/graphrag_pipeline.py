try:
    from .graphrag_retriever import GraphRAGRetriever
except ImportError:
    from graphrag_retriever import GraphRAGRetriever

from ollama import chat
retriever = GraphRAGRetriever()


def build_graph_context(result):
    """
    Convert TigerGraph relationships into readable evidence.
    """

    if not result["found"]:
        return "No matching merchant was found."

    lines = []

    lines.append(
        f"Merchant: {result['merchant']}"
    )

    lines.append("\nConnected merchants:")

    for relationship in result["relationships"]:
        target = relationship["target"]
        weight = relationship["attributes"].get("weight")

        lines.append(
            f"- {target} (relationship weight: {weight})"
        )

    return "\n".join(lines)


def graphrag_pipeline(question, merchant_name):

    # 1. Retrieve graph evidence
    result = retriever.search_merchant(
        merchant_name
    )

    # 2. Build evidence context
    context = build_graph_context(result)

    # 3. Ask LLM
    prompt = f"""
You are a GraphRAG assistant.

Answer the user's question using ONLY the graph evidence.

User question:
{question}

Graph evidence:
{context}

Rules:
- Do not invent information.
- Do not remove relevant entities from the evidence.
- Preserve the merchant names exactly.
- If there are 10 connected merchants, mention all 10.
- Keep the answer concise.
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

    # 4. Return structured result
    return {
        "question": question,
        "answer": answer,
        "evidence": context,
        "evidence_count": len(result["relationships"])
    }


if __name__ == "__main__":

    question = (
        "What merchants are connected to "
        "Hahn, Bahringer and McLaughlin?"
    )

    merchant_name = (
        "Hahn, Bahringer and McLaughlin"
    )

    result = graphrag_pipeline(
        question,
        merchant_name
    )

    print("\nQuestion:")
    print(result["question"])

    print("\nGraphRAG Answer:")
    print(result["answer"])

    print("\nEvidence Count:")
    print(result["evidence_count"])

    print("\nGraph Evidence:")
    print(result["evidence"])