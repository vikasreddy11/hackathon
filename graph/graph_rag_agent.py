# ============================================================
# graph/graph_rag_agent.py
# ============================================================

from graph.graphrag_retriever import GraphRAGRetriever


class GraphRAGAgent:
    """
    Agent layer for the GraphRAG + TigerGraph fraud system.

    Example:

        agent = GraphRAGAgent()
        result = agent.answer("show fraudulent transactions")
        print(result)
    """

    def __init__(self):
        self.retriever = GraphRAGRetriever()
        print("GraphRAG Agent initialized.")

    # ========================================================
    # ANSWER QUERY
    # ========================================================

    def answer(self, question):
        """
        Retrieve relevant graph information and return
        a clean human-readable answer.
        """

        if not question or not str(question).strip():
            return "Please provide a fraud-related query."

        question = str(question).strip()

        try:
            result = self.retriever.search(question)

            if not result:
                return "No result was returned."

            return self._format_result(result)

        except Exception as e:
            return (
                f"Agent error: {type(e).__name__}: {e}"
            )

    # ========================================================
    # FORMAT RESULT
    # ========================================================

    def _format_result(self, result):

        result_type = result.get(
            "type",
            "unknown"
        )

        # ====================================================
        # GENERAL FRAUD
        # ====================================================

        if result_type == "fraud_transactions":

            records = result.get(
                "records",
                []
            )

            if not records:
                return (
                    "No fraudulent transactions found."
                )

            lines = [
                f"Found {len(records)} "
                f"fraudulent transactions:"
            ]

            for record in records:

                lines.append(
                    self._format_transaction(
                        record
                    )
                )

            return "\n".join(lines)

        # ====================================================
        # TOP FRAUD
        # ====================================================

        if result_type == "top_fraud":

            records = result.get(
                "records",
                []
            )

            if not records:
                return (
                    "No fraudulent transactions found."
                )

            lines = [
                "Highest-value fraudulent transactions:"
            ]

            for index, record in enumerate(
                records,
                start=1
            ):

                lines.append(
                    f"{index}. "
                    f"{self._format_transaction(record)}"
                )

            return "\n".join(lines)

        # ====================================================
        # HIGH VALUE FRAUD
        # ====================================================

        if result_type == "high_value_fraud":

            records = result.get(
                "records",
                []
            )

            threshold = result.get(
                "threshold",
                result.get(
                    "minimum_amount"
                )
            )

            total_count = result.get(
                "total_count",
                result.get(
                    "count",
                    len(records)
                )
            )

            if threshold is None:

                threshold_text = ""

            else:

                threshold_text = (
                    f" above ${float(threshold):,.2f}"
                )

            if not records:

                return (
                    f"No fraudulent transactions found"
                    f"{threshold_text}."
                )

            lines = [
                f"Found {total_count} fraudulent "
                f"transactions{threshold_text}:"
            ]

            for record in records:

                lines.append(
                    self._format_transaction(
                        record
                    )
                )

            return "\n".join(lines)

        # ====================================================
        # TRANSACTION INVESTIGATION
        # ====================================================

        if result_type == "transaction_investigation":

            transaction_id = result.get(
                "transaction_id",
                "requested transaction"
            )

            transaction = result.get(
                "transaction"
            )

            connected = result.get(
                "connected_transactions",
                []
            )

            # ------------------------------------------------
            # Header
            # ------------------------------------------------

            lines = [
                f"Transaction Investigation: "
                f"{transaction_id}"
            ]

            lines.append("")

            # ------------------------------------------------
            # Main transaction
            # ------------------------------------------------

            if transaction:

                lines.append(
                    self._format_transaction(
                        transaction
                    )
                )

            else:

                lines.append(
                    "Transaction details were not found."
                )

            # ------------------------------------------------
            # No connected transactions
            # ------------------------------------------------

            if not connected:

                lines.append("")
                lines.append(
                    "Graph evidence:"
                )

                lines.append(
                    "- No connected transactions were found."
                )

                lines.append("")
                lines.append(
                    "Assessment: "
                    "No graph-based fraud evidence was found "
                    "for this transaction."
                )

                return "\n".join(lines)

            # ------------------------------------------------
            # Count graph relationships
            # ------------------------------------------------

            shared_card = 0
            shared_merchant = 0
            shared_both = 0
            fraudulent_connected = 0

            for record in connected:

                connection = str(
                    record.get(
                        "connection",
                        ""
                    )
                ).strip().lower()

                if connection == "shared_card":

                    shared_card += 1

                elif connection == "shared_merchant":

                    shared_merchant += 1

                elif (
                    connection
                    == "shared_card_and_merchant"
                ):

                    shared_both += 1

                fraud = str(
                    record.get(
                        "is_fraud",
                        ""
                    )
                ).strip().lower()

                if fraud in {
                    "1",
                    "1.0",
                    "true",
                    "yes"
                }:

                    fraudulent_connected += 1

            # ------------------------------------------------
            # Graph evidence summary
            # ------------------------------------------------

            lines.append("")

            lines.append(
                "Graph evidence:"
            )

            lines.append(
                f"- Connected transactions: "
                f"{len(connected)}"
            )

            lines.append(
                f"- Shared card connections: "
                f"{shared_card}"
            )

            lines.append(
                f"- Shared merchant connections: "
                f"{shared_merchant}"
            )

            lines.append(
                f"- Shared card + merchant connections: "
                f"{shared_both}"
            )

            lines.append(
                f"- Connected fraudulent transactions: "
                f"{fraudulent_connected}"
            )

            # ------------------------------------------------
            # Investigation assessment
            # ------------------------------------------------

            transaction_fraud = str(
                transaction.get(
                    "is_fraud",
                    ""
                )
                if transaction
                else ""
            ).strip().lower()

            lines.append("")

            lines.append(
                "Assessment:"
            )

            # Main transaction itself is fraud
            if transaction_fraud in {
                "1",
                "1.0",
                "true",
                "yes"
            }:

                lines.append(
                    "The transaction itself is marked as fraud."
                )

                if fraudulent_connected > 0:

                    lines.append(
                        f"{fraudulent_connected} connected "
                        f"fraudulent transaction(s) were also "
                        f"found in the retrieved graph neighborhood."
                    )

            # Main transaction isn't fraud but connected
            # fraudulent transactions exist
            elif fraudulent_connected > 0:

                lines.append(
                    "The transaction is not marked as fraud, "
                    "but fraudulent transactions exist in its "
                    "connected graph neighborhood."
                )

                lines.append(
                    "These graph relationships provide "
                    "additional evidence that should be "
                    "investigated further."
                )

            # No fraud anywhere in neighborhood
            else:

                lines.append(
                    "No strong fraud evidence was found "
                    "in the retrieved graph neighborhood."
                )

                lines.append(
                    "The transaction has graph connections, "
                    "but the retrieved connected transactions "
                    "are also marked as non-fraud."
                )

            # ------------------------------------------------
            # Connected transaction evidence
            # ------------------------------------------------

            lines.append("")

            lines.append(
                "Connected transaction evidence:"
            )

            for index, record in enumerate(
                connected[:10],
                start=1
            ):

                connection = record.get(
                    "connection",
                    "connected"
                )

                lines.append(
                    f"{index}. "
                    f"{self._format_transaction(record)} "
                    f"[{connection}]"
                )

            return "\n".join(lines)

        # ====================================================
        # SINGLE TRANSACTION
        # ====================================================

        if result_type == "transaction":

            transaction = result.get(
                "transaction"
            )

            if not transaction:

                transaction_id = result.get(
                    "transaction_id",
                    "requested transaction"
                )

                return (
                    f"Transaction {transaction_id} "
                    f"was not found."
                )

            return (
                "Transaction details:\n"
                + self._format_transaction(
                    transaction
                )
            )

        # ====================================================
        # CONNECTED TRANSACTIONS
        # ====================================================

        if result_type == "connected_transactions":

            transaction_id = result.get(
                "transaction_id",
                "requested transaction"
            )

            records = result.get(
                "records",
                []
            )

            if not records:

                return (
                    f"No connected transactions found "
                    f"for transaction {transaction_id}."
                )

            lines = [
                f"Transactions connected to "
                f"transaction {transaction_id}:"
            ]

            for index, record in enumerate(
                records,
                start=1
            ):

                connection = record.get(
                    "connection",
                    "connected"
                )

                transaction_text = (
                    self._format_transaction(
                        record
                    )
                )

                lines.append(
                    f"{index}. "
                    f"{transaction_text} "
                    f"[{connection}]"
                )

            return "\n".join(lines)

        # ====================================================
        # MERCHANT
        # ====================================================

        if result_type == "merchant":

            merchant = result.get(
                "merchant",
                "requested merchant"
            )

            records = result.get(
                "records",
                []
            )

            if not records:

                return (
                    f"No transactions found for "
                    f"merchant {merchant}."
                )

            lines = [
                f"Transactions for merchant "
                f"{merchant}:"
            ]

            for record in records:

                lines.append(
                    self._format_transaction(
                        record
                    )
                )

            return "\n".join(lines)

        # ====================================================
        # MERCHANT FRAUD
        # ====================================================

        if result_type == "merchant_fraud":

            merchant = result.get(
                "merchant",
                "requested merchant"
            )

            fraud_result = result.get(
                "records",
                []
            )

            # Newer retriever may return:
            #
            # {
            #     "records": [...],
            #     "total_count": ...
            # }
            #
            # Older retriever returns a list.

            if isinstance(
                fraud_result,
                dict
            ):

                records = fraud_result.get(
                    "records",
                    []
                )

                total_count = fraud_result.get(
                    "total_count",
                    len(records)
                )

            else:

                records = fraud_result

                total_count = result.get(
                    "total_count",
                    result.get(
                        "count",
                        len(records)
                    )
                )

            if not records:

                return (
                    f"No fraudulent transactions "
                    f"found for merchant {merchant}."
                )

            lines = [
                f"Found {total_count} fraudulent "
                f"transactions for merchant "
                f"{merchant}:"
            ]

            for record in records:

                lines.append(
                    self._format_transaction(
                        record
                    )
                )

            return "\n".join(lines)

        # ====================================================
        # UNKNOWN
        # ====================================================

        if result_type == "unknown":

            return result.get(
                "answer",
                "I could not determine the "
                "requested fraud query."
            )

        # ====================================================
        # FALLBACK
        # ====================================================

        return self._format_generic(
            result
        )

    # ========================================================
    # FORMAT TRANSACTION
    # ========================================================

    @staticmethod
    def _format_transaction(record):

        transaction_id = record.get(
            "transaction_id",
            record.get(
                "v_id",
                record.get(
                    "id",
                    "unknown"
                )
            )
        )

        amount = record.get(
            "amount"
        )

        fraud = record.get(
            "is_fraud"
        )

        transaction_time = record.get(
            "transaction_time"
        )

        # ----------------------------------------------------
        # Amount
        # ----------------------------------------------------

        if amount is None:

            amount_text = "N/A"

        else:

            try:

                amount_text = (
                    f"${float(amount):,.2f}"
                )

            except (
                TypeError,
                ValueError
            ):

                amount_text = str(
                    amount
                )

        # ----------------------------------------------------
        # Fraud status
        # ----------------------------------------------------

        fraud_text = (
            "Fraud"
            if str(
                fraud
            ).strip().lower()
            in {
                "1",
                "1.0",
                "true",
                "yes"
            }
            else "Not fraud"
        )

        return (
            f"Transaction {transaction_id} | "
            f"Amount: {amount_text} | "
            f"Status: {fraud_text} | "
            f"Time: "
            f"{transaction_time or 'N/A'}"
        )

    # ========================================================
    # GENERIC FALLBACK
    # ========================================================

    @staticmethod
    def _format_generic(result):

        records = result.get(
            "records",
            []
        )

        if records:

            lines = [
                f"Found {len(records)} result(s):"
            ]

            for record in records:

                if isinstance(
                    record,
                    dict
                ):

                    lines.append(
                        GraphRAGAgent._format_transaction(
                            record
                        )
                    )

                else:

                    lines.append(
                        str(record)
                    )

            return "\n".join(lines)

        answer = result.get(
            "answer"
        )

        if answer:

            return str(
                answer
            )

        return str(
            result
        )