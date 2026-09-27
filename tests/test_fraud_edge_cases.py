from graph.graphrag_retriever import GraphRAGRetriever


retriever = GraphRAGRetriever()


def test_zero_threshold():
    result = retriever.search(
        "show fraudulent transactions above 0"
    )

    assert result["type"] == "high_value_fraud"
    assert result["threshold"] == 0.0

    for record in result["records"]:
        assert float(record["amount"]) > 0


def test_extremely_high_threshold():
    result = retriever.search(
        "show fraudulent transactions above 1000000"
    )

    assert result["type"] == "high_value_fraud"
    assert result["threshold"] == 1000000.0
    assert result["count"] == 0


def test_nonexistent_transaction():
    result = retriever.search(
        "show transaction 999999999"
    )

    assert result["type"] == "transaction"
    assert result["count"] == 0


def test_nonexistent_connected_transaction():
    result = retriever.search(
        "which transactions are connected to transaction 999999999"
    )

    assert result["type"] == "connected_transactions"
    assert result["count"] == 0


def test_unknown_query():
    result = retriever.search(
        "tell me something completely unrelated"
    )

    assert result["type"] == "unknown"
    assert result["count"] == 0