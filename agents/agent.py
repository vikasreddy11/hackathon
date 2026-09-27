
# ============================================================
# agents/agent.py
# ============================================================

import sys
import re

from graph.graphrag_retriever import GraphRAGRetriever

from agents.response_formatter import (
    format_agent_response
)


# ============================================================
# MERCHANT QUERY DETECTION
# ============================================================

def is_merchant_query(question):
    """
    Detect whether the user is asking about
    transactions belonging to a specific merchant.
    """

    q = question.lower().strip()

    if "transaction" not in q:
        return False

    patterns = [
        "merchant",
        "for merchant",
        "merchant transactions",
        "transactions for",
        "transactions involving",
        "transactions associated with",
        "transactions linked to",
        "from merchant",
        "at merchant"
    ]

    return any(
        pattern in q
        for pattern in patterns
    )


# ============================================================
# GRAPH RAG FRAUD AGENT
# ============================================================

class GraphRAGAgent:

    def __init__(self):

        self.retriever = GraphRAGRetriever()

        print(
            "GraphRAG Fraud Agent initialized."
        )

    # ========================================================
    # DECISION
    # ========================================================

    def decide(self, question):

        q = question.lower().strip()

        # ----------------------------------------------------
        # MERCHANT QUERY
        # ----------------------------------------------------

        if is_merchant_query(question):
            return "merchant"

        # ----------------------------------------------------
        # CONNECTED TRANSACTIONS
        # ----------------------------------------------------

        if (
            "connected" in q
            or "same card" in q
            or "same merchant" in q
            or "linked" in q
            or "associated" in q
            or "related" in q
            or "involving" in q
        ):

            if re.search(
                r"\btransaction\s+\d+\b",
                q
            ):

                return "connected_transactions"

        # ----------------------------------------------------
        # SINGLE TRANSACTION
        # ----------------------------------------------------

        if "transaction" in q:

            if re.search(
                r"\btransaction\s+\d+\b",
                q
            ):

                return "transaction"

        # ----------------------------------------------------
        # HIGH VALUE FRAUD
        # ----------------------------------------------------

        if (
            any(
                word in q
                for word in [
                    "above",
                    "over",
                    "greater than",
                    "more than"
                ]
            )
            and
            any(
                word in q
                for word in [
                    "fraud",
                    "fraudulent",
                    "suspicious"
                ]
            )
        ):

            return "high_value_fraud"

        # ----------------------------------------------------
        # TOP FRAUD
        # ----------------------------------------------------

        if (
            any(
                word in q
                for word in [
                    "highest",
                    "largest",
                    "top",
                    "maximum",
                    "most expensive"
                ]
            )
            and
            any(
                word in q
                for word in [
                    "fraud",
                    "fraudulent",
                    "suspicious"
                ]
            )
        ):

            return "top_fraud"

        # ----------------------------------------------------
        # GENERAL FRAUD
        # ----------------------------------------------------

        if any(
            word in q
            for word in [
                "fraud",
                "fraudulent",
                "suspicious"
            ]
        ):

            return "fraud_transactions"

        # ----------------------------------------------------
        # UNKNOWN
        # ----------------------------------------------------

        return "unknown"

    # ========================================================
    # NORMALIZE RESULT
    # ========================================================

    def normalize_result(self, result):

        # ----------------------------------------------------
        # INVALID RESULT
        # ----------------------------------------------------

        if not isinstance(result, dict):

            return {
                "type": "unknown",
                "records": [],
                "count": 0,
                "total_count": 0
            }

        # ----------------------------------------------------
        # NORMALIZE RECORDS
        # ----------------------------------------------------

        records = result.get(
            "records",
            []
        )

        if not isinstance(records, list):
            records = []

        result["records"] = records

        result_type = result.get(
            "type",
            "unknown"
        )

        result["type"] = result_type

        # ====================================================
        # HIGH VALUE FRAUD
        # ====================================================

        if result_type == "high_value_fraud":

            total_count = result.get(
                "total_count"
            )

            if total_count is None:
                total_count = result.get(
                    "count"
                )

            if total_count is None:
                total_count = len(records)

            try:
                total_count = int(total_count)

            except (
                TypeError,
                ValueError
            ):
                total_count = len(records)

            # Never allow total count to be lower
            # than the records actually returned.
            if total_count < len(records):
                total_count = len(records)

            result["total_count"] = total_count
            result["count"] = total_count

            return result

        # ====================================================
        # MERCHANT FRAUD
        # ====================================================

        if result_type == "merchant_fraud":

            total_count = result.get(
                "total_count"
            )

            if total_count is None:
                total_count = result.get(
                    "count"
                )

            if total_count is None:
                total_count = len(records)

            try:
                total_count = int(total_count)

            except (
                TypeError,
                ValueError
            ):
                total_count = len(records)

            if total_count < len(records):
                total_count = len(records)

            result["total_count"] = total_count
            result["count"] = total_count

            return result

        # ====================================================
        # ALL OTHER RESULT TYPES
        # ====================================================

        count = result.get(
            "count"
        )

        try:
            count = int(count)

        except (
            TypeError,
            ValueError
        ):
            count = len(records)

        if count < len(records):
            count = len(records)

        result["count"] = count

        # Keep total_count available where possible.
        if "total_count" not in result:
            result["total_count"] = count

        return result

    # ========================================================
    # HIGH VALUE FALLBACK FORMATTER
    # ========================================================

    def format_high_value_fallback(self, result):

        records = result.get(
            "records",
            []
        )

        minimum_amount = result.get(
            "minimum_amount",
            0
        )

        total_count = result.get(
            "total_count",
            len(records)
        )

        # ----------------------------------------------------
        # NORMALIZE MINIMUM AMOUNT
        # ----------------------------------------------------

        try:
            minimum_amount = float(
                minimum_amount
            )

        except (
            TypeError,
            ValueError
        ):
            minimum_amount = 0

        # ----------------------------------------------------
        # NORMALIZE COUNT
        # ----------------------------------------------------

        try:
            total_count = int(
                total_count
            )

        except (
            TypeError,
            ValueError
        ):
            total_count = len(records)

        # ----------------------------------------------------
        # NO RESULTS
        # ----------------------------------------------------

        if total_count == 0:

            return (
                "No fraudulent transactions found "
                f"above ${minimum_amount:,.2f}."
            )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        lines = [
            (
                f"Found {total_count:,} fraudulent "
                f"transactions above "
                f"${minimum_amount:,.2f}."
            )
        ]

        if records:

            lines.append(
                f"Showing {len(records)} "
                f"matching transactions:"
            )

        # ----------------------------------------------------
        # RECORDS
        # ----------------------------------------------------

        for index, record in enumerate(
            records,
            start=1
        ):

            if not isinstance(
                record,
                dict
            ):
                continue

            transaction_id = (
                record.get(
                    "transaction_id"
                )
                or
                record.get(
                    "v_id"
                )
                or
                record.get(
                    "id"
                )
                or
                "Unknown"
            )

            try:

                amount = float(
                    record.get(
                        "amount",
                        0
                    ) or 0
                )

            except (
                TypeError,
                ValueError
            ):

                amount = 0

            transaction_time = record.get(
                "transaction_time"
            )

            lines.append(
                f"\n{index}. Transaction: "
                f"{transaction_id}"
            )

            lines.append(
                f"   Amount: "
                f"${amount:,.2f}"
            )

            lines.append(
                "   Status: Fraudulent"
            )

            if transaction_time:

                lines.append(
                    f"   Time: "
                    f"{transaction_time}"
                )

            merchant = record.get(
                "merchant"
            )

            if merchant:

                lines.append(
                    f"   Merchant: "
                    f"{merchant}"
                )

        return "\n".join(lines)

    # ========================================================
    # MERCHANT FRAUD FALLBACK FORMATTER
    # ========================================================

    def format_merchant_fraud_fallback(
        self,
        result
    ):

        records = result.get(
            "records",
            []
        )

        merchant = result.get(
            "merchant",
            "requested merchant"
        )

        total_count = result.get(
            "total_count",
            len(records)
        )

        try:

            total_count = int(
                total_count
            )

        except (
            TypeError,
            ValueError
        ):

            total_count = len(records)

        # ----------------------------------------------------
        # NO RESULTS
        # ----------------------------------------------------

        if total_count == 0:

            return (
                f"No fraudulent transactions found "
                f"for merchant '{merchant}'."
            )

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        lines = [
            (
                f"Found {total_count:,} fraudulent "
                f"transactions for merchant "
                f"'{merchant}'."
            )
        ]

        if records:

            lines.append(
                f"Showing {len(records)} "
                f"matching transactions:"
            )

        # ----------------------------------------------------
        # RECORDS
        # ----------------------------------------------------

        for index, record in enumerate(
            records,
            start=1
        ):

            if not isinstance(
                record,
                dict
            ):
                continue

            transaction_id = (
                record.get(
                    "transaction_id"
                )
                or
                record.get(
                    "v_id"
                )
                or
                record.get(
                    "id"
                )
                or
                "Unknown"
            )

            try:

                amount = float(
                    record.get(
                        "amount",
                        0
                    ) or 0
                )

            except (
                TypeError,
                ValueError
            ):

                amount = 0

            transaction_time = record.get(
                "transaction_time"
            )

            lines.append(
                f"\n{index}. Transaction: "
                f"{transaction_id}"
            )

            lines.append(
                f"   Amount: "
                f"${amount:,.2f}"
            )

            lines.append(
                "   Status: Fraudulent"
            )

            if transaction_time:

                lines.append(
                    f"   Time: "
                    f"{transaction_time}"
                )

            record_merchant = record.get(
                "merchant"
            )

            if record_merchant:

                lines.append(
                    f"   Merchant: "
                    f"{record_merchant}"
                )

        return "\n".join(lines)

    # ========================================================
    # GENERIC RESULT FALLBACK
    # ========================================================

    def format_generic_fallback(
        self,
        result
    ):

        records = result.get(
            "records",
            []
        )

        result_type = result.get(
            "type",
            "unknown"
        )

        count = result.get(
            "count",
            len(records)
        )

        if not records:

            return (
                f"No matching data found "
                f"for query type '{result_type}'."
            )

        lines = [
            (
                f"Found {count:,} matching "
                f"records."
            ),
            (
                f"Showing {len(records)} records:"
            )
        ]

        for index, record in enumerate(
            records,
            start=1
        ):

            if not isinstance(
                record,
                dict
            ):
                continue

            transaction_id = (
                record.get(
                    "transaction_id"
                )
                or
                record.get(
                    "v_id"
                )
                or
                record.get(
                    "id"
                )
            )

            amount = record.get(
                "amount"
            )

            merchant = record.get(
                "merchant"
            )

            transaction_time = record.get(
                "transaction_time"
            )

            lines.append(
                f"\n{index}."
            )

            if transaction_id:
                lines.append(
                    f"   Transaction: "
                    f"{transaction_id}"
                )

            if amount is not None:

                try:

                    amount_value = float(
                        amount
                    )

                    lines.append(
                        f"   Amount: "
                        f"${amount_value:,.2f}"
                    )

                except (
                    TypeError,
                    ValueError
                ):

                    lines.append(
                        f"   Amount: {amount}"
                    )

            if merchant:
                lines.append(
                    f"   Merchant: "
                    f"{merchant}"
                )

            if transaction_time:
                lines.append(
                    f"   Time: "
                    f"{transaction_time}"
                )

        return "\n".join(lines)

    # ========================================================
    # RUN AGENT
    # ========================================================

    def run(self, question):

        print()
        print(
            "========== AGENT =========="
        )

        print(
            "Question:",
            question
        )

        # ----------------------------------------------------
        # DECIDE QUERY TYPE
        # ----------------------------------------------------

        decision = self.decide(
            question
        )

        print(
            "Decision:",
            decision
        )

        # ----------------------------------------------------
        # RETRIEVE DATA
        # ----------------------------------------------------

        try:

            result = self.retriever.search(
                question
            )

        except Exception as e:

            print(
                "Retriever error:",
                type(e).__name__,
                ":",
                e
            )

            result = {
                "type": "unknown",
                "records": [],
                "count": 0,
                "total_count": 0,
                "error": str(e)
            }

        # ----------------------------------------------------
        # NORMALIZE DATA
        # ----------------------------------------------------

        result = self.normalize_result(
            result
        )

        # ----------------------------------------------------
        # DEBUG INFORMATION
        # ----------------------------------------------------

        print(
            "Result type:",
            result.get(
                "type"
            )
        )

        print(
            "Count:",
            result.get(
                "count",
                0
            )
        )

        print(
            "Returned records:",
            len(
                result.get(
                    "records",
                    []
                )
            )
        )

        if result.get(
            "type"
        ) == "high_value_fraud":

            print(
                "Total matching transactions:",
                result.get(
                    "total_count",
                    0
                )
            )

            try:

                minimum_amount = float(
                    result.get(
                        "minimum_amount",
                        0
                    ) or 0
                )

            except (
                TypeError,
                ValueError
            ):

                minimum_amount = 0

            print(
                "Minimum amount:",
                f"${minimum_amount:,.2f}"
            )

        # ----------------------------------------------------
        # TRY NORMAL FORMATTER
        # ----------------------------------------------------

        try:

            answer = format_agent_response(
                question,
                result
            )

        except Exception as e:

            print(
                "Response formatting error:",
                type(e).__name__,
                ":",
                e
            )

            answer = None

        # ====================================================
        # HIGH VALUE FRAUD FALLBACK
        # ====================================================

        result_type = result.get(
            "type"
        )

        if result_type == "high_value_fraud":

            total_count = result.get(
                "total_count",
                0
            )

            if (
                answer is None
                or
                (
                    total_count > 0
                    and
                    (
                        not result.get(
                            "records"
                        )
                        or
                        answer.strip()
                        .lower()
                        .startswith(
                            "no fraudulent transactions"
                        )
                    )
                )
            ):

                answer = (
                    self.format_high_value_fallback(
                        result
                    )
                )

        # ====================================================
        # MERCHANT FRAUD FALLBACK
        # ====================================================

        elif result_type == "merchant_fraud":

            records = result.get(
                "records",
                []
            )

            if records:

                if (
                    answer is None
                    or
                    answer.strip()
                    .lower()
                    .startswith(
                        "no fraudulent transactions"
                    )
                ):

                    answer = (
                        self.format_merchant_fraud_fallback(
                            result
                        )
                    )

            elif result.get(
                "total_count",
                0
            ) == 0:

                if answer is None:

                    answer = (
                        f"No fraudulent transactions "
                        f"found for merchant "
                        f"'{result.get('merchant', 'requested merchant')}'."
                    )

        # ====================================================
        # MERCHANT RESULT FALLBACK
        # ====================================================

        elif result_type == "merchant":

            if answer is None:

                merchant = result.get(
                    "merchant",
                    "requested merchant"
                )

                records = result.get(
                    "records",
                    []
                )

                if records:

                    answer = (
                        f"Found "
                        f"{len(records)} "
                        f"transactions for merchant "
                        f"'{merchant}'."
                    )

                else:

                    answer = (
                        f"No transaction data found "
                        f"for merchant "
                        f"'{merchant}'."
                    )

        # ====================================================
        # GENERAL / OTHER RESULT FALLBACK
        # ====================================================

        if answer is None:

            answer = self.format_generic_fallback(
                result
            )

        # ----------------------------------------------------
        # FINAL ANSWER
        # ----------------------------------------------------

        print()
        print(
            "========== FINAL ANSWER =========="
        )

        print(
            answer
        )

        return {
            "answer": answer,
            "evidence": result,
            "question_type": decision
        }


# ============================================================
# MAIN
# ============================================================

def main():

    question = " ".join(
        sys.argv[1:]
    ).strip()

    if not question:

        question = (
            "show fraudulent transactions"
        )

    agent = GraphRAGAgent()

    agent.run(
        question
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
