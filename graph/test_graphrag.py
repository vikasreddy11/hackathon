from graphrag_retriever import GraphRAGRetriever


retriever = GraphRAGRetriever()


question = "What merchants are connected to Hahn, Bahringer and McLaughlin?"


merchant_name = "Hahn, Bahringer and McLaughlin"


result = retriever.search_merchant(merchant_name)


print("\nQuestion:")
print(question)

print("\nGraphRAG Evidence:")

if not result["found"]:
    print("Merchant not found.")

else:
    print("Merchant:", result["merchant"])

    print("\nConnected merchants:")

    for relationship in result["relationships"]:
        print(
            "-",
            relationship["target"],
            "| weight:",
            relationship["attributes"].get("weight")
        )