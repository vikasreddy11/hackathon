# ============================================================
# tests/test_fraud_agent.py
# ============================================================

from graph.graphrag_retriever import GraphRAGRetriever


retriever = GraphRAGRetriever()


def test_general_fraud():
    result = retriever.search(
        "show fraudulent transactions"
    )

    assert result["type"] == "fraud_transactions"
    assert result["count"] > 0
    assert len(result["records"]) > 0


def test_fraud_above_500():
    result = retriever.search(
        "show fraudulent transactions above 500"
    )

    assert result["type"] == "high_value_fraud"
    assert result["threshold"] == 500.0
    assert result["count"] > 0

    for record in result["records"]:
        assert float(record["amount"]) > 500


def test_fraud_above_1000():
    result = retriever.search(
        "show fraud over 1000"
    )

    assert result["type"] == "high_value_fraud"
    assert result["threshold"] == 1000.0

    for record in result["records"]:
        assert float(record["amount"]) > 1000


def test_top_fraud():
    result = retriever.search(
        "which fraudulent transactions have the highest amounts"
    )

    assert result["type"] == "top_fraud"
    assert result["count"] > 0

    amounts = [
        float(record["amount"])
        for record in result["records"]
    ]

    assert amounts == sorted(
        amounts,
        reverse=True
    )


def test_transaction_lookup():
    result = retriever.search(
        "show transaction 14971"
    )

    assert result["type"] == "transaction"
    assert result["count"] == 1

    transaction = result["transaction"]

    assert transaction is not None
    assert str(
        transaction["transaction_id"]
    ) == "14971"


def test_connected_transactions():
    result = retriever.search(
        "which transactions are connected to transaction 14971"
    )

    assert result["type"] == "connected_transactions"
    assert result["transaction_id"] == "14971"
    assert result["count"] > 0

    for record in result["records"]:
        assert "transaction_id" in record
        assert "amount" in record


print("All fraud-agent tests loaded.")