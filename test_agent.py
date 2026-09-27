from graph.graph_rag_agent import GraphRAGAgent


def main():
    agent = GraphRAGAgent()

    queries = [
        "show fraudulent transactions",
        "show the highest fraud transactions",
        "show transaction 123",
        "show transactions connected to transaction 123",
    ]

    for query in queries:
        print("\n" + "=" * 70)
        print("QUERY:", query)
        print("=" * 70)

        try:
            result = agent.answer(query)
            print(result)

        except Exception as e:
            print("ERROR:", type(e).__name__, ":", e)


if __name__ == "__main__":
    main()