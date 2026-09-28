# ============================================================
# graph/graphrag_retriever.py
# ============================================================

import re

from graph.tigergraph_search import (
    search_fraud_transactions,
    search_high_value_fraud,
    get_top_fraud_transactions,
    get_transaction,
    get_connected_transactions,
    get_merchant,
    get_merchant_transactions,
    get_merchant_fraud_transactions,
)


class GraphRAGRetriever:

    def __init__(self):
        print("GraphRAG Retriever initialized.")

    # ========================================================
    # AMOUNT EXTRACTION
    # ========================================================

    def extract_amount(self, question):
        """
        Extract monetary amount from queries such as:

        $500
        $1,000
        500
        500.50
        above $500
        over 1000
        greater than $250.75
        """

        q = str(question).lower().strip()

        # Remove commas from numbers
        q = q.replace(",", "")

        patterns = [
            # $500 / $500.50
            r"\$\s*(\d+(?:\.\d+)?)",

            # USD 500
            r"\busd\s*(\d+(?:\.\d+)?)",

            # 500 dollars / 500 dollar
            r"(\d+(?:\.\d+)?)\s*dollars?\b",

            # above 500 / over 500 / greater than 500
            r"(?:above|over|greater\s+than|more\s+than|exceeding)\s+(\d+(?:\.\d+)?)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                q
            )

            if match:

                try:
                    return float(
                        match.group(1)
                    )

                except (
                    TypeError,
                    ValueError
                ):
                    pass

        return None

    # ========================================================
    # TRANSACTION ID EXTRACTION
    # ========================================================

    def extract_transaction_id(self, question):

        q = str(
            question
        ).lower().strip()

        match = re.search(
            r"\btransaction\s+(\d+)\b",
            q
        )

        if match:
            return match.group(1)

        return None

    # ========================================================
    # MERCHANT NAME EXTRACTION
    # ========================================================

    def extract_merchant_name(self, question):

        q = str(
            question
        ).lower().strip()

        patterns = [

            # transactions for merchant Walmart
            r"transactions?\s+for\s+merchant\s+(.+)",

            # transactions from merchant Walmart
            r"transactions?\s+from\s+merchant\s+(.+)",

            # transactions at merchant Walmart
            r"transactions?\s+at\s+merchant\s+(.+)",

            # transactions involving merchant Walmart
            r"transactions?\s+involving\s+merchant\s+(.+)",

            # merchant Walmart transactions
            r"merchant\s+(.+?)\s+transactions?",

            # for merchant Walmart
            r"for\s+merchant\s+(.+)",

            # from merchant Walmart
            r"from\s+merchant\s+(.+)",

            # at merchant Walmart
            r"at\s+merchant\s+(.+)",
        ]

        for pattern in patterns:

            match = re.search(
                pattern,
                q
            )

            if match:

                merchant = (
                    match.group(1)
                    .strip()
                    .rstrip("?.!,")
                )

                # Don't treat generic "merchant" as a real merchant.
                if merchant in (
                    "",
                    "merchant",
                    "the merchant"
                ):
                    return None

                return merchant

        return None

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        question
    ):

        if not question:

            return {
                "type": "unknown",
                "records": [],
                "count": 0,
                "total_count": 0,
                "answer":
                    "Please provide a fraud-related query."
            }

        q = str(
            question
        ).lower().strip()

        print(
            "DEBUG query:",
            question
        )

        # ====================================================
        # 0. TRANSACTION INVESTIGATION
        # ====================================================

        investigation_words = [
            "why",
            "suspicious",
            "investigate",
            "investigation",
            "explain",
            "reason",
            "evidence",
        ]

        transaction_id = (
            self.extract_transaction_id(
                question
            )
        )

        if (
            transaction_id
            and
            "transaction" in q
            and
            any(
                word in q
                for word in investigation_words
            )
        ):

            print(
                "DEBUG transaction investigation:",
                transaction_id
            )

            transaction = get_transaction(
                transaction_id
            )

            connected = (
                get_connected_transactions(
                    transaction_id,
                    limit=20
                )
            )

            if not isinstance(
                connected,
                list
            ):
                connected = []

            records = []

            # Put original transaction first.
            if transaction:
                records.append(
                    transaction
                )

            records.extend(
                connected
            )

            return {
                "type":
                    "transaction_investigation",

                "transaction_id":
                    transaction_id,

                "transaction":
                    transaction,

                "connected_transactions":
                    connected,

                "records":
                    records,

                "connected_count":
                    len(connected),

                "count":
                    len(records),

                "total_count":
                    len(records)
            }

        # ====================================================
        # 1. SPECIFIC TRANSACTION
        # ====================================================

        if transaction_id:

            # Connected transaction query
            if any(
                word in q
                for word in [
                    "connected",
                    "associated",
                    "linked",
                    "related",
                    "same merchant",
                    "same card",
                    "involving",
                ]
            ):

                print(
                    "DEBUG connected transaction:",
                    transaction_id
                )

                records = (
                    get_connected_transactions(
                        transaction_id,
                        limit=20
                    )
                )

                if not isinstance(
                    records,
                    list
                ):
                    records = []

                return {
                    "type":
                        "connected_transactions",

                    "transaction_id":
                        transaction_id,

                    "records":
                        records,

                    "count":
                        len(records),

                    "total_count":
                        len(records)
                }

            # Normal transaction query
            print(
                "DEBUG transaction:",
                transaction_id
            )

            transaction = get_transaction(
                transaction_id
            )

            records = (
                [transaction]
                if transaction
                else []
            )

            return {
                "type":
                    "transaction",

                "transaction":
                    transaction,

                "records":
                    records,

                "count":
                    len(records),

                "total_count":
                    len(records)
            }

        # ====================================================
        # 2. HIGH VALUE FRAUD
        # ====================================================

        amount = self.extract_amount(
            question
        )

        has_fraud_word = any(
            word in q
            for word in [
                "fraud",
                "fraudulent",
                "suspicious"
            ]
        )

        has_threshold_word = any(
            phrase in q
            for phrase in [
                "above",
                "over",
                "greater than",
                "more than",
                "exceeding"
            ]
        )

        if (
            amount is not None
            and has_fraud_word
            and has_threshold_word
        ):

            print(
                "DEBUG HIGH VALUE FRAUD"
            )

            print(
                "DEBUG minimum amount:",
                f"${amount:,.2f}"
            )

            search_result = (
                search_high_value_fraud(
                    minimum_amount=amount,
                    limit=20
                )
            )

            if not isinstance(
                search_result,
                dict
            ):

                search_result = {}

            records = search_result.get(
                "records",
                []
            )

            if not isinstance(
                records,
                list
            ):
                records = []

            total_count = search_result.get(
                "total_count"
            )

            if total_count is None:
                total_count = search_result.get(
                    "count"
                )

            if total_count is None:
                total_count = len(records)

            try:
                total_count = int(
                    total_count
                )
            except (
                TypeError,
                ValueError
            ):
                total_count = len(records)

            print(
                "DEBUG high-value records:",
                len(records)
            )

            print(
                "DEBUG high-value total:",
                total_count
            )

            return {
                "type":
                    "high_value_fraud",

                "minimum_amount":
                    amount,

                "records":
                    records,

                "total_count":
                    total_count,

                "count":
                    total_count
            }

        # ====================================================
        # 3. MERCHANT QUERY
        # ====================================================

        merchant_name = (
            self.extract_merchant_name(
                question
            )
        )

        if merchant_name:

            print(
                "DEBUG merchant name:",
                repr(merchant_name)
            )

            merchant = get_merchant(
                merchant_name
            )

            print(
                "DEBUG merchant result:",
                merchant
            )

            if merchant is None:

                return {
                    "type":
                        "merchant",

                    "merchant":
                        merchant_name,

                    "records":
                        [],

                    "count":
                        0,

                    "total_count":
                        0
                }

            # ------------------------------------------------
            # Merchant fraud
            # ------------------------------------------------

            if any(
                word in q
                for word in [
                    "fraud",
                    "fraudulent",
                    "suspicious"
                ]
            ):

                records = (
                    get_merchant_fraud_transactions(
                        merchant_name,
                        limit=20
                    )
                )

                if not isinstance(
                    records,
                    list
                ):
                    records = []

                return {
                    "type":
                        "merchant_fraud",

                    "merchant":
                        merchant.get(
                            "merchant",
                            merchant_name
                        ),

                    "merchant_id":
                        merchant.get(
                            "merchant_id"
                        ),

                    "records":
                        records,

                    "count":
                        len(records),

                    "total_count":
                        len(records)
                }

            # ------------------------------------------------
            # All merchant transactions
            # ------------------------------------------------

            records = (
                get_merchant_transactions(
                    merchant_name,
                    limit=20
                )
            )

            if not isinstance(
                records,
                list
            ):
                records = []

            return {
                "type":
                    "merchant",

                "merchant":
                    merchant.get(
                        "merchant",
                        merchant_name
                    ),

                "merchant_id":
                    merchant.get(
                        "merchant_id"
                    ),

                "records":
                    records,

                "count":
                    len(records),

                "total_count":
                    len(records)
            }

        # ====================================================
        # 4. TOP FRAUD
        # ====================================================

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
            and has_fraud_word
        ):

            print(
                "DEBUG TOP FRAUD"
            )

            records = (
                get_top_fraud_transactions(
                    limit=10
                )
            )

            if not isinstance(
                records,
                list
            ):
                records = []

            return {
                "type":
                    "top_fraud",

                "records":
                    records,

                "count":
                    len(records),

                "total_count":
                    len(records)
            }

        # ====================================================
        # 5. GENERAL FRAUD
        # ====================================================

        if has_fraud_word:

            print(
                "DEBUG GENERAL FRAUD"
            )

            records = (
                search_fraud_transactions(
                    limit=20
                )
            )

            if not isinstance(
                records,
                list
            ):
                records = []

            return {
                "type":
                    "fraud_transactions",

                "records":
                    records,

                "count":
                    len(records),

                "total_count":
                    len(records)
            }

        # ====================================================
        # 6. UNKNOWN
        # ====================================================

        return {
            "type":
                "unknown",

            "records":
                [],

            "count":
                0,

            "total_count":
                0,

            "answer":
                "I could not determine the requested fraud query."
        }

    # ========================================================
    # ALIASES
    # ========================================================

    def retrieve(
        self,
        question
    ):

        return self.search(
            question
        )

    def run(
        self,
        question
    ):

        return self.search(
            question
        )