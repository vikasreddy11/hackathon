
# ============================================================
# agents/response_formatter.py
# ============================================================

from typing import (
    Any,
    Dict,
    List,
    Optional
)


# ============================================================
# RECORD HELPERS
# ============================================================

def get_records(
    result: Any
) -> List[Dict[str, Any]]:

    if not result:
        return []

    if isinstance(result, dict):

        records = result.get("records")

        if isinstance(records, list):

            return [
                r
                for r in records
                if isinstance(r, dict)
            ]

        record = result.get("record")

        if isinstance(record, dict):

            return [record]

    return []


# ============================================================
# TRANSACTION ID
# ============================================================

def transaction_id(record):

    return (
        record.get("transaction_id")
        or record.get("v_id")
        or record.get("id")
        or "unknown"
    )


# ============================================================
# MONEY
# ============================================================

def money(value):

    try:
        return f"${float(value):,.2f}"

    except (
        TypeError,
        ValueError
    ):
        return "$0.00"


# ============================================================
# FRAUD LABEL
# ============================================================

def fraud_label(record):

    value = str(
        record.get("is_fraud")
    ).lower()

    if value in [
        "1",
        "1.0",
        "true"
    ]:
        return "Fraudulent"

    if value in [
        "0",
        "0.0",
        "false"
    ]:
        return "Not fraudulent"

    # Some TigerGraph queries already return
    # only fraudulent records.
    if record.get("is_fraud") is None:
        return "Fraudulent"

    return "Unknown"


# ============================================================
# TRANSACTION FORMAT
# ============================================================

def format_transaction(
    record,
    index: Optional[int] = None
):

    prefix = (
        f"{index}. "
        if index
        else ""
    )

    lines = [

        (
            f"{prefix}Transaction: "
            f"{transaction_id(record)}"
        ),

        (
            f"   Amount: "
            f"{money(record.get('amount'))}"
        ),

        (
            f"   Status: "
            f"{fraud_label(record)}"
        ),
    ]

    if record.get(
        "transaction_time"
    ) is not None:

        lines.append(
            (
                f"   Time: "
                f"{record['transaction_time']}"
            )
        )

    if record.get(
        "merchant"
    ) is not None:

        lines.append(
            (
                f"   Merchant: "
                f"{record['merchant']}"
            )
        )

    if record.get(
        "connected_merchant"
    ) is not None:

        lines.append(
            (
                f"   Connected merchant: "
                f"{record['connected_merchant']}"
            )
        )

    if record.get(
        "connected_card"
    ) is not None:

        lines.append(
            (
                f"   Connected card: "
                f"{record['connected_card']}"
            )
        )

    if record.get(
        "connection"
    ) is not None:

        lines.append(
            (
                f"   Connection: "
                f"{record['connection']}"
            )
        )

    return "\n".join(lines)


# ============================================================
# SINGLE TRANSACTION
# ============================================================

def format_transaction_lookup(
    result: Dict[str, Any]
) -> str:

    records = get_records(result)

    if not records:

        return "Transaction not found."

    return format_transaction(
        records[0]
    )


# ============================================================
# MERCHANT TRANSACTIONS
# ============================================================

def format_merchant_transactions(
    result: Dict[str, Any]
) -> str:

    records = get_records(result)

    merchant = result.get("merchant")

    if not records:

        return (
            f"No transaction data found "
            f"for merchant '{merchant}'."
        )

    record = records[0]

    # Merchant statistics result
    if "transaction_count" in record:

        return (
            f"Merchant: {merchant}\n"
            f"Transactions: "
            f"{record['transaction_count']:,}\n"
            f"Total amount: "
            f"${float(record['total_amount']):,.2f}\n"
            f"Average amount: "
            f"${float(record['average_amount']):,.2f}\n"
            f"Minimum amount: "
            f"${float(record['minimum_amount']):,.2f}\n"
            f"Maximum amount: "
            f"${float(record['maximum_amount']):,.2f}"
        )

    lines = [
        (
            f"Found {len(records)} transactions "
            f"for merchant '{merchant}'."
        ),
        ""
    ]

    for index, record in enumerate(
        records[:20],
        start=1
    ):

        lines.append(
            format_transaction(
                record,
                index
            )
        )

        lines.append("")

    return "\n".join(lines).rstrip()


# ============================================================
# MERCHANT FRAUD
# ============================================================

def format_merchant_fraud(
    result: Dict[str, Any]
) -> str:

    records = get_records(result)

    merchant = result.get("merchant")

    if not records:

        return (
            f"No fraudulent transactions "
            f"found for merchant "
            f"'{merchant}'."
        )

    # Use total_count if retriever provides it.
    total_count = result.get(
        "total_count",
        len(records)
    )

    lines = [

        (
            f"Found {total_count:,} "
            f"fraudulent transactions "
            f"for merchant "
            f"'{merchant}'."
        ),

        (
            f"Showing {len(records)} "
            f"matching transactions:"
        ),

        ""
    ]

    for index, record in enumerate(
        records[:20],
        start=1
    ):

        lines.append(
            format_transaction(
                record,
                index
            )
        )

        lines.append("")

    return "\n".join(lines).rstrip()


# ============================================================
# FRAUD TRANSACTIONS
# ============================================================

def format_fraud_transactions(
    result
):

    records = get_records(result)

    if not records:

        return (
            "No fraudulent transactions found."
        )

    total_count = result.get(
        "total_count",
        len(records)
    )

    lines = [

        (
            f"Found {total_count:,} "
            f"fraudulent transactions."
        ),

        (
            f"Showing {len(records)} "
            f"transactions:"
        ),

        ""
    ]

    for index, record in enumerate(
        records[:20],
        start=1
    ):

        lines.append(
            format_transaction(
                record,
                index
            )
        )

        lines.append("")

    return "\n".join(lines).rstrip()


# ============================================================
# HIGH VALUE FRAUD
# ============================================================

def format_high_value_fraud(
    result
):

    records = get_records(result)

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = result.get(
        "threshold",
        result.get(
            "minimum_amount",
            0
        )
    )

    try:
        threshold = float(threshold)

    except (
        TypeError,
        ValueError
    ):
        threshold = 0.0

    # --------------------------------------------------------
    # Total matching transactions
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # No results
    # --------------------------------------------------------

    if total_count == 0:

        return (
            f"No fraudulent transactions "
            f"found above "
            f"${threshold:,.2f}."
        )

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    lines = [

        (
            f"Found {total_count:,} "
            f"fraudulent transactions "
            f"above "
            f"${threshold:,.2f}."
        ),

        (
            f"Showing {len(records)} "
            f"matching transactions:"
        ),

        ""
    ]

    # --------------------------------------------------------
    # Records
    # --------------------------------------------------------

    for index, record in enumerate(
        records[:20],
        start=1
    ):

        lines.append(
            format_transaction(
                record,
                index
            )
        )

        lines.append("")

    return "\n".join(
        lines
    ).rstrip()


# ============================================================
# TOP FRAUD
# ============================================================

def format_top_fraud(
    result
):

    records = get_records(result)

    if not records:

        return (
            "No fraudulent transactions found."
        )

    total_count = result.get(
        "total_count",
        len(records)
    )

    lines = [

        (
            f"Top {len(records)} "
            f"fraudulent transactions "
            f"by amount:"
        ),

        ""
    ]

    for index, record in enumerate(
        records[:20],
        start=1
    ):

        lines.append(
            format_transaction(
                record,
                index
            )
        )

        lines.append("")

    return "\n".join(
        lines
    ).rstrip()


# ============================================================
# CONNECTED TRANSACTIONS
# ============================================================

def format_connected_transactions(
    result
):

    records = get_records(result)

    transaction = result.get(
        "transaction_id"
    )

    if not records:

        return (
            f"No connected transactions "
            f"found for transaction "
            f"{transaction}."
        )

    total_count = result.get(
        "total_count",
        len(records)
    )

    lines = [

        (
            f"Found {total_count:,} "
            f"transactions connected to "
            f"transaction {transaction}."
        ),

        (
            f"Showing {len(records)} "
            f"connected transactions:"
        ),

        ""
    ]

    for index, record in enumerate(
        records[:20],
        start=1
    ):

        lines.append(
            format_transaction(
                record,
                index
            )
        )

        lines.append("")

    return "\n".join(
        lines
    ).rstrip()


# ============================================================
# UNKNOWN
# ============================================================

def format_unknown(
    result
):

    return str(
        result.get(
            "answer",
            "I could not find an answer."
        )
    )


# ============================================================
# MAIN FORMAT ROUTER
# ============================================================

def format_agent_response(
    question: str,
    result: Optional[
        Dict[str, Any]
    ]
) -> str:

    if not result:

        return (
            "I could not find an answer."
        )

    result_type = result.get(
        "type"
    )

    # --------------------------------------------------------
    # MERCHANT
    # --------------------------------------------------------

    if result_type == "merchant":

        return format_merchant_transactions(
            result
        )

    # --------------------------------------------------------
    # MERCHANT FRAUD
    # --------------------------------------------------------

    if result_type == "merchant_fraud":

        return format_merchant_fraud(
            result
        )

    # --------------------------------------------------------
    # FRAUD
    # --------------------------------------------------------

    if result_type == "fraud_transactions":

        return format_fraud_transactions(
            result
        )

    # --------------------------------------------------------
    # HIGH VALUE FRAUD
    # --------------------------------------------------------

    if result_type == "high_value_fraud":

        return format_high_value_fraud(
            result
        )

    # --------------------------------------------------------
    # TOP FRAUD
    # --------------------------------------------------------

    if result_type == "top_fraud":

        return format_top_fraud(
            result
        )

    # --------------------------------------------------------
    # CONNECTED
    # --------------------------------------------------------

    if result_type == "connected_transactions":

        return format_connected_transactions(
            result
        )

    # --------------------------------------------------------
    # TRANSACTION
    # --------------------------------------------------------

    if result_type == "transaction":

        return format_transaction_lookup(
            result
        )

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    if result_type == "unknown":

        return format_unknown(
            result
        )

    # --------------------------------------------------------
    # GENERIC FALLBACK
    # --------------------------------------------------------

    records = get_records(result)

    if records:

        total_count = result.get(
            "total_count",
            len(records)
        )

        lines = [

            f"Found {total_count:,} records.",

            f"Showing {len(records)} records:",

            ""
        ]

        for index, record in enumerate(
            records[:20],
            start=1
        ):

            lines.append(
                format_transaction(
                    record,
                    index
                )
            )

            lines.append("")

        return "\n".join(
            lines
        ).rstrip()

    return format_unknown(
        result
    )