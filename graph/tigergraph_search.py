# ============================================================
# graph/tigergraph_search.py
# ============================================================

import re

from graph.tigergraph_client import get_tigergraph_connection


# ============================================================
# COMPACT TRANSACTION
# ============================================================

def compact_transaction(record):
    if not record:
        return None

    return {
        "transaction_id": record.get("v_id"),
        "amount": record.get("amount"),
        "is_fraud": record.get("is_fraud"),
        "transaction_time": record.get("transaction_time"),
    }


# ============================================================
# FRAUD VALUE CHECK
# ============================================================

def is_fraudulent(value):
    if value is None:
        return False

    return str(value).strip().lower() in {
        "1",
        "1.0",
        "true",
        "yes",
    }


# ============================================================
# AMOUNT EXTRACTION
# ============================================================

def extract_amount(text):
    if not text:
        return None

    match = re.search(
        r'(?:\$|usd\s*)\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
        text.lower()
    )

    if not match:
        match = re.search(
            r'\b(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:dollars|usd)\b',
            text.lower()
        )

    if not match:
        return None

    try:
        return float(
            match.group(1).replace(",", "")
        )
    except ValueError:
        return None

# ============================================================
# GET ALL TRANSACTIONS
# ============================================================

def get_all_transactions(limit=20):
    conn = get_tigergraph_connection()

    try:
        print(
            f"DEBUG fetching up to {limit} transactions"
        )

        df = conn.getVertexDataFrame(
            "Payment_Transaction",
            select=(
                "amount,"
                "is_fraud,"
                "transaction_time"
            ),
            limit=limit,
            timeout=30000,
        )

        if df is None or df.empty:
            return []

        records = []

        for record in df.to_dict(orient="records"):
            compact = compact_transaction(record)

            if compact:
                records.append(compact)

        return records

    except Exception as e:
        print(
            "Transaction search error:",
            type(e).__name__,
            ":",
            e,
        )

        return []


# ============================================================
# FRAUD TRANSACTIONS
# ============================================================

def search_fraud_transactions(limit=20):
    conn = get_tigergraph_connection()

    try:
        print(
            f"DEBUG fraud query: "
            f"fetching up to {limit} fraudulent transactions"
        )

        df = conn.getVertexDataFrame(
            "Payment_Transaction",
            select=(
                "amount,"
                "is_fraud,"
                "transaction_time"
            ),
            where="is_fraud=1",
            limit=limit,
            timeout=30000,
        )

        if df is None or df.empty:
            print(
                "DEBUG no fraudulent transactions returned"
            )
            return []

        records = []

        for record in df.to_dict(orient="records"):
            compact = compact_transaction(record)

            if compact:
                records.append(compact)

        print(
            "DEBUG fraudulent transactions returned:",
            len(records),
        )

        return records

    except Exception as e:
        print(
            "Fraud transaction search error:",
            type(e).__name__,
            ":",
            e,
        )

        return []


# ============================================================
# SINGLE TRANSACTION
# ============================================================

def get_transaction(transaction_id):
    conn = get_tigergraph_connection()

    try:
        df = conn.getVertexDataFrameById(
            "Payment_Transaction",
            str(transaction_id),
        )

        if df is None or df.empty:
            return None

        record = df.iloc[0].to_dict()

        return compact_transaction(record)

    except Exception as e:
        print(
            "DEBUG get_transaction ERROR:",
            type(e).__name__,
            ":",
            e,
        )

        return None


# ============================================================
# HIGH VALUE FRAUD
# ============================================================

def search_high_value_fraud(minimum_amount, limit=20):
    conn = get_tigergraph_connection()

    try:
        minimum_amount = float(minimum_amount)
        limit = int(limit)

        print(
            f"DEBUG high-value fraud minimum amount: "
            f"${minimum_amount:,.2f}"
        )

        count_df = conn.getVertexDataFrame(
            "Payment_Transaction",
            select="amount",
            where=f"is_fraud=1,amount>={minimum_amount}",
            limit=100000,
            timeout=30000,
        )

        total_count = (
            0
            if count_df is None
            else len(count_df)
        )

        df = conn.getVertexDataFrame(
            "Payment_Transaction",
            select=(
                "amount,"
                "is_fraud,"
                "transaction_time"
            ),
            where=f"is_fraud=1,amount>={minimum_amount}",
            sort="-amount",
            limit=limit,
            timeout=30000,
        )

        if df is None or df.empty:
            print(
                f"DEBUG fraudulent transactions >= "
                f"${minimum_amount:,.2f}: 0"
            )

            return {
                "records": [],
                "total_count": total_count,
            }

        records = []

        for record in df.to_dict(
            orient="records"
        ):
            compact = compact_transaction(record)

            if compact:
                records.append(compact)

        print(
            f"DEBUG fraudulent transactions >= "
            f"${minimum_amount:,.2f}: "
            f"{total_count}"
        )

        return {
            "records": records,
            "total_count": total_count,
        }

    except Exception as e:
        print(
            "ERROR search_high_value_fraud:",
            type(e).__name__,
            str(e),
        )

        return {
            "records": [],
            "total_count": 0,
        }


# ============================================================
# TOP FRAUD TRANSACTIONS
# ============================================================

def get_top_fraud_transactions(limit=10):
    conn = get_tigergraph_connection()

    try:
        print(
            f"DEBUG fetching top {limit} "
            f"fraudulent transactions"
        )

        df = conn.getVertexDataFrame(
            "Payment_Transaction",
            select=(
                "amount,"
                "is_fraud,"
                "transaction_time"
            ),
            where="is_fraud=1",
            sort="-amount",
            limit=limit,
            timeout=30000,
        )

        if df is None or df.empty:
            print(
                "DEBUG no top fraud transactions found"
            )
            return []

        records = []

        for record in df.to_dict(orient="records"):
            compact = compact_transaction(record)

            if compact:
                records.append(compact)

        print(
            "DEBUG top fraud transactions returned:",
            len(records),
        )

        return records

    except Exception as e:
        print(
            "Top fraud search error:",
            type(e).__name__,
            ":",
            e,
        )

        return []


# ============================================================
# CONNECTED TRANSACTIONS
# ============================================================

def get_connected_transactions(
    transaction_id,
    limit=10,
):
    conn = get_tigergraph_connection()

    transaction_id = str(transaction_id)

    connected = {}

    # --------------------------------------------------------
    # Transaction -> Merchant
    # --------------------------------------------------------

    try:
        merchant_edges = conn.getEdges(
            "Payment_Transaction",
            transaction_id,
            "Merchant_Receive_Transaction",
        )
    except Exception as e:
        print(
            "Merchant traversal error:",
            e,
        )
        merchant_edges = []

    merchant_ids = []

    for edge in merchant_edges or []:
        merchant_id = edge.get("to_id")

        if merchant_id is not None:
            merchant_ids.append(
                str(merchant_id)
            )

    # --------------------------------------------------------
    # Transaction -> Card
    # --------------------------------------------------------

    try:
        card_edges = conn.getEdges(
            "Payment_Transaction",
            transaction_id,
            "Card_Send_Transaction",
        )
    except Exception as e:
        print(
            "Card traversal error:",
            e,
        )
        card_edges = []

    card_ids = []

    for edge in card_edges or []:
        card_id = edge.get("to_id")

        if card_id is not None:
            card_ids.append(
                str(card_id)
            )

    # --------------------------------------------------------
    # Merchant -> Transactions
    # --------------------------------------------------------

    for merchant_id in merchant_ids:

        try:
            edges = conn.getEdges(
                "Merchant",
                merchant_id,
                "Merchant_Receive_Transaction",
            )
        except Exception as e:
            print(
                "Merchant reverse traversal error:",
                e,
            )
            continue

        for edge in edges or []:

            txn_id = edge.get("to_id")

            if txn_id is None:
                continue

            txn_id = str(txn_id)

            if txn_id == transaction_id:
                continue

            connected.setdefault(
                txn_id,
                {
                    "transaction_id": txn_id,
                    "connection": "shared_merchant",
                    "merchant_id": merchant_id,
                },
            )

    # --------------------------------------------------------
    # Card -> Transactions
    # --------------------------------------------------------

    for card_id in card_ids:

        try:
            edges = conn.getEdges(
                "Card",
                card_id,
                "Card_Send_Transaction",
            )
        except Exception as e:
            print(
                "Card reverse traversal error:",
                e,
            )
            continue

        for edge in edges or []:

            txn_id = edge.get("to_id")

            if txn_id is None:
                continue

            txn_id = str(txn_id)

            if txn_id == transaction_id:
                continue

            if txn_id in connected:

                connected[txn_id][
                    "connection"
                ] = "shared_card_and_merchant"

                connected[txn_id][
                    "card_id"
                ] = card_id

            else:

                connected[txn_id] = {
                    "transaction_id": txn_id,
                    "connection": "shared_card",
                    "card_id": card_id,
                }

    # --------------------------------------------------------
    # Fetch transaction details
    # --------------------------------------------------------

    results = []

    for txn_id, info in list(
        connected.items()
    )[:limit]:

        try:
            df = conn.getVertexDataFrameById(
                "Payment_Transaction",
                txn_id,
            )

            if df is None or df.empty:
                continue

            record = df.iloc[0].to_dict()

            transaction = compact_transaction(
                record
            )

            if transaction is None:
                continue

            transaction["connection"] = info.get(
                "connection",
                "connected",
            )

            if "merchant_id" in info:
                transaction[
                    "connected_merchant"
                ] = info["merchant_id"]

            if "card_id" in info:
                transaction[
                    "connected_card"
                ] = info["card_id"]

            results.append(transaction)

        except Exception as e:
            print(
                f"Connected transaction {txn_id} "
                f"lookup error:",
                e,
            )

    return results


# ============================================================
# SPECIAL TRANSACTION INVESTIGATION
# ============================================================

def investigate_transaction(transaction_id):

    c = get_tigergraph_connection()

    transaction_id = str(transaction_id)

    tx = get_transaction(transaction_id)

    if not tx:
        return {
            "type": "transaction_investigation",
            "found": False,
            "transaction_id": transaction_id,
            "transaction": None,
            "merchant_id": None,
            "card_id": None,
            "evidence": [],
            "answer": (
                f"Transaction {transaction_id} "
                "was not found in the graph."
            ),
        }

    evidence = []

    # ========================================================
    # 1. FRAUD LABEL
    # ========================================================

    if is_fraudulent(tx.get("is_fraud")):

        evidence.append({
            "type": "fraud_label",
            "description": (
                "Transaction is labelled as fraudulent "
                "in the graph."
            ),
            "value": 1,
        })

    # ========================================================
    # 2. FIND MERCHANT AND CARD
    # ========================================================

    try:
        edges = c.getEdges(
            "Payment_Transaction",
            transaction_id,
        )
    except Exception as e:
        print(
            "Transaction edge lookup error:",
            e,
        )
        edges = []

    merchant_id = None
    card_id = None

    for edge in edges or []:

        edge_type = edge.get("e_type")

        if edge_type == "Merchant_Receive_Transaction":
            merchant_id = edge.get("to_id")

        elif edge_type == "Card_Send_Transaction":
            card_id = edge.get("to_id")

    # ========================================================
    # 3. MERCHANT EVIDENCE
    # ========================================================

    if merchant_id:

        try:

            merchant_edges = c.getEdges(
                "Merchant",
                merchant_id,
            )

        except Exception as e:

            print(
                "Merchant evidence lookup error:",
                e,
            )

            merchant_edges = []

        merchant_connections = [
            e
            for e in merchant_edges
            if e.get("e_type")
            == "Merchant_Merchant"
        ]

        merchant_cards = [
            e
            for e in merchant_edges
            if e.get("e_type")
            == "Has_Interaction_With_Merchant"
            and e.get("to_type") == "Card"
        ]

        merchant_community = [
            e
            for e in merchant_edges
            if e.get("e_type")
            == "Has_Community"
        ]

        evidence.append({
            "type": "merchant",
            "merchant_id": merchant_id,
            "merchant_connections": len(
                merchant_connections
            ),
            "connected_cards": len(
                merchant_cards
            ),
            "communities": [
                e.get("to_id")
                for e in merchant_community
            ],
        })

    # ========================================================
    # 4. CARD EVIDENCE
    # ========================================================

    if card_id:

        try:

            card_edges = c.getEdges(
                "Card",
                card_id,
            )

        except Exception as e:

            print(
                "Card evidence lookup error:",
                e,
            )

            card_edges = []

        card_merchants = [
            e
            for e in card_edges
            if e.get("e_type")
            == "Has_Interaction_With_Merchant"
            and e.get("to_type") == "Merchant"
        ]

        card_transactions = [
            e
            for e in card_edges
            if e.get("e_type")
            == "Card_Send_Transaction"
            and e.get("to_type")
            == "Payment_Transaction"
        ]

        evidence.append({
            "type": "card",
            "card_id": card_id,
            "merchant_count": len(
                card_merchants
            ),
            "transaction_count": len(
                card_transactions
            ),
        })

    # ========================================================
    # 5. CONNECTED TRANSACTIONS
    # ========================================================

    connected = get_connected_transactions(
        transaction_id,
        limit=10,
    )

    fraud_connected = [
        record
        for record in connected
        if is_fraudulent(
            record.get("is_fraud")
        )
    ]

    evidence.append({
        "type": "connected_transactions",
        "connected_count": len(
            connected
        ),
        "connected_fraud_count": len(
            fraud_connected
        ),
        "fraud_transactions": (
            fraud_connected[:20]
        ),
    })

    # ========================================================
    # 6. BUILD INVESTIGATION RESULT
    # ========================================================

    investigation = {
        "type": "transaction_investigation",
        "found": True,
        "transaction": tx,
        "merchant_id": merchant_id,
        "card_id": card_id,
        "evidence": evidence,
    }

    # ========================================================
    # 7. GENERATE EXPLANATION
    # ========================================================

    investigation["answer"] = (
        explain_transaction_investigation(
            investigation
        )
    )

    return investigation
# ============================================================
# TRANSACTION INVESTIGATION EXPLANATION
# ============================================================

def explain_transaction_investigation(investigation):
    """
    Convert structured transaction investigation data
    into a human-readable fraud explanation.
    """

    if not investigation:
        return "No investigation result was returned."

    if not investigation.get("found"):
        return (
            f"Transaction {investigation.get('transaction_id')} "
            "was not found in the graph."
        )

    tx = investigation.get("transaction", {})
    evidence = investigation.get("evidence", [])

    transaction_id = tx.get("transaction_id")
    amount = tx.get("amount")
    fraud = tx.get("is_fraud")
    transaction_time = tx.get("transaction_time")

    lines = []

    lines.append(
        f"Transaction {transaction_id} is classified as "
        f"{'fraudulent' if int(fraud or 0) == 1 else 'not fraudulent'}."
    )

    if amount is not None:
        lines.append(
            f"Transaction amount: ${float(amount):,.2f}."
        )

    if transaction_time:
        lines.append(
            f"Transaction time: {transaction_time}."
        )

    for item in evidence:

        evidence_type = item.get("type")

        if evidence_type == "fraud_label":

            lines.append(
                "The transaction has an explicit fraud label "
                "in the TigerGraph dataset."
            )

        elif evidence_type == "merchant":

            merchant_id = item.get("merchant_id")
            merchant_connections = item.get(
                "merchant_connections",
                0
            )
            connected_cards = item.get(
                "connected_cards",
                0
            )
            communities = item.get(
                "communities",
                []
            )

            lines.append(
                f"It is associated with merchant "
                f"{merchant_id}."
            )

            lines.append(
                f"The merchant has {merchant_connections} "
                f"merchant connections and "
                f"{connected_cards} connected cards."
            )

            if communities:
                lines.append(
                    f"The merchant belongs to community "
                    f"{', '.join(map(str, communities))}."
                )

        elif evidence_type == "card":

            card_id = item.get("card_id")
            merchant_count = item.get(
                "merchant_count",
                0
            )
            transaction_count = item.get(
                "transaction_count",
                0
            )

            lines.append(
                f"The transaction was made using card "
                f"{card_id}."
            )

            lines.append(
                f"This card is connected to "
                f"{merchant_count} merchants and "
                f"{transaction_count} transactions "
                f"in the graph."
            )

        elif evidence_type == "connected_transactions":

            connected_count = item.get(
                "connected_count",
                0
            )
            connected_fraud_count = item.get(
                "connected_fraud_count",
                0
            )

            lines.append(
                f"The graph found {connected_count} "
                f"connected transactions."
            )

            lines.append(
                f"{connected_fraud_count} of those "
                f"connected transactions are labelled fraudulent."
            )

    return "\n".join(lines)

# ============================================================
# MERCHANT LOOKUP
# ============================================================

def get_merchant(merchant_name):
    """
    Resolve a merchant name to its TigerGraph Merchant ID.

    Supports:
        fraud_Baumbach Ltd
        Baumbach Ltd
        Baumbach
    """

    if not merchant_name:
        return None

    merchant_name = str(
        merchant_name
    ).strip()

    if not merchant_name:
        return None

    conn = get_tigergraph_connection()

    try:

        merchants = conn.getVertices(
            "Merchant",
            limit=10000,
        )

        target = merchant_name.casefold()

        partial_matches = []

        for merchant in merchants or []:

            attributes = merchant.get(
                "attributes",
                {},
            )

            merchant_id = merchant.get(
                "v_id"
            )

            if merchant_id is None:
                merchant_id = merchant.get(
                    "id"
                )

            if merchant_id is None:
                continue

            merchant_id = str(
                merchant_id
            )

            merchant_display = merchant_id

            if merchant_display.startswith(
                "fraud_"
            ):
                merchant_display = (
                    merchant_display[6:]
                )

            display_lower = (
                merchant_display.casefold()
            )

            id_lower = (
                merchant_id.casefold()
            )

            # Exact match
            if (
                display_lower == target
                or id_lower == target
            ):

                print(
                    "DEBUG merchant exact match:",
                    merchant_id,
                )

                return {
                    "merchant_id": merchant_id,
                    "merchant": merchant_display,
                    "attributes": attributes,
                }

            # Partial match
            if target in display_lower:

                partial_matches.append(
                    {
                        "merchant_id": merchant_id,
                        "merchant": merchant_display,
                        "attributes": attributes,
                    }
                )

        if partial_matches:

            print(
                "DEBUG merchant partial match:",
                partial_matches[0][
                    "merchant_id"
                ],
            )

            return partial_matches[0]

        print(
            f"DEBUG merchant not found: "
            f"'{merchant_name}'"
        )

    except Exception as e:

        print(
            "Merchant lookup error:",
            type(e).__name__,
            ":",
            e,
        )

    return None


# ============================================================
# MERCHANT FRAUD SUMMARY
# ============================================================

def get_merchant_fraud_summary(merchant_id):

    conn = get_tigergraph_connection()

    merchant = get_merchant(
        merchant_id
    )

    if not merchant:

        return {
            "error": (
                f"Merchant not found: "
                f"{merchant_id}"
            )
        }

    edges = conn.getEdges(
        "Merchant",
        merchant["merchant_id"],
        "Merchant_Receive_Transaction",
    )

    transaction_ids = [
        edge["to_id"]
        for edge in edges
        if edge.get("to_type")
        == "Payment_Transaction"
    ]

    if not transaction_ids:

        return {
            "merchant_id": merchant[
                "merchant_id"
            ],
            "merchant": merchant[
                "merchant"
            ],
            "transaction_count": 0,
            "fraud_count": 0,
            "fraud_rate": 0.0,
            "total_amount": 0.0,
            "fraud_amount": 0.0,
            "pagerank": merchant[
                "attributes"
            ].get("pagerank"),
            "community_id": merchant[
                "attributes"
            ].get("c_id"),
            "community_size": merchant[
                "attributes"
            ].get("c_size"),
        }

    batch_size = 100
    rows = []

    for i in range(
        0,
        len(transaction_ids),
        batch_size,
    ):

        batch_ids = transaction_ids[
            i:i + batch_size
        ]

        df = conn.getVertexDataFrameById(
            "Payment_Transaction",
            batch_ids,
            select=(
                "amount,"
                "is_fraud,"
                "transaction_time,"
                "mer_cat,"
                "age,"
                "gender"
            ),
        )

        if df is not None and not df.empty:
            rows.extend(
                df.to_dict(
                    "records"
                )
            )

    transaction_count = len(rows)

    fraud_count = sum(
        1
        for row in rows
        if is_fraudulent(
            row.get("is_fraud")
        )
    )

    total_amount = sum(
        float(
            row.get("amount", 0) or 0
        )
        for row in rows
    )

    fraud_amount = sum(
        float(
            row.get("amount", 0) or 0
        )
        for row in rows
        if is_fraudulent(
            row.get("is_fraud")
        )
    )

    fraud_rate = (
        (fraud_count / transaction_count)
        * 100
        if transaction_count
        else 0.0
    )

    return {
        "merchant_id": merchant[
            "merchant_id"
        ],
        "merchant": merchant[
            "merchant"
        ],
        "transaction_count": transaction_count,
        "fraud_count": fraud_count,
        "fraud_rate": round(
            fraud_rate,
            2,
        ),
        "total_amount": round(
            total_amount,
            2,
        ),
        "fraud_amount": round(
            fraud_amount,
            2,
        ),
        "pagerank": merchant[
            "attributes"
        ].get("pagerank"),
        "community_id": merchant[
            "attributes"
        ].get("c_id"),
        "community_size": merchant[
            "attributes"
        ].get("c_size"),
    }


# ============================================================
# MERCHANT TRANSACTIONS
# ============================================================

def get_merchant_transactions(
    merchant_name,
    limit=20,
):

    print(
        f"DEBUG merchant name: "
        f"{merchant_name}"
    )

    merchant = get_merchant(
        merchant_name
    )

    if merchant is None:

        print(
            "DEBUG merchant result: None"
        )

        return []

    canonical_id = merchant[
        "merchant_id"
    ]

    print(
        "DEBUG canonical merchant ID:",
        canonical_id,
    )

    conn = get_tigergraph_connection()

    try:

        edges = conn.getEdges(
            "Merchant",
            canonical_id,
            "Merchant_Receive_Transaction",
        )

        print(
            "DEBUG merchant transaction edges:",
            len(edges or []),
        )

        if not edges:
            return []

        transaction_ids = []

        for edge in edges:

            txn_id = edge.get(
                "to_id"
            )

            if txn_id is not None:
                transaction_ids.append(
                    str(txn_id)
                )

        print(
            "DEBUG fetching merchant transactions:",
            len(transaction_ids),
        )

        vertices = conn.getVerticesById(
            "Payment_Transaction",
            transaction_ids,
        )

        records = []

        for vertex in vertices or []:

            attributes = vertex.get(
                "attributes",
                {},
            )

            records.append(
                {
                    "transaction_id": vertex.get(
                        "v_id"
                    ),
                    "amount": attributes.get(
                        "amount"
                    ),
                    "is_fraud": attributes.get(
                        "is_fraud"
                    ),
                    "transaction_time": attributes.get(
                        "transaction_time"
                    ),
                    "merchant": merchant[
                        "merchant"
                    ],
                }
            )

            if len(records) >= limit:
                break

        print(
            "DEBUG merchant transactions found:",
            len(records),
        )

        return records

    except Exception as e:

        print(
            "Merchant transaction query error:",
            type(e).__name__,
            ":",
            e,
        )

        return []


# ============================================================
# MERCHANT FRAUD TRANSACTIONS
# ============================================================

def get_merchant_fraud_transactions(
    merchant_name,
    limit=20,
):

    print(
        f"DEBUG merchant fraud search: "
        f"{merchant_name}"
    )

    merchant = get_merchant(
        merchant_name
    )

    if merchant is None:

        print(
            "DEBUG merchant result: None"
        )

        return {
            "records": [],
            "total_count": 0,
        }

    canonical_id = merchant[
        "merchant_id"
    ]

    print(
        "DEBUG canonical merchant ID:",
        canonical_id,
    )

    conn = get_tigergraph_connection()

    try:

        edges = conn.getEdges(
            "Merchant",
            canonical_id,
            "Merchant_Receive_Transaction",
        )

        transaction_ids = [
            str(edge.get("to_id"))
            for edge in (edges or [])
            if edge.get("to_id") is not None
        ]

        print(
            "DEBUG merchant transaction edges:",
            len(transaction_ids),
        )

        if not transaction_ids:

            return {
                "records": [],
                "total_count": 0,
            }

        batch_size = 100
        fraud_records = []

        for start in range(
            0,
            len(transaction_ids),
            batch_size,
        ):

            batch = transaction_ids[
                start:start + batch_size
            ]

            print(
                f"DEBUG fetching batch "
                f"{start + 1}-"
                f"{min(start + batch_size, len(transaction_ids))}"
            )

            try:

                vertices = conn.getVerticesById(
                    "Payment_Transaction",
                    batch,
                )

            except Exception as e:

                print(
                    "DEBUG batch error:",
                    type(e).__name__,
                    ":",
                    e,
                )

                continue

            for vertex in vertices or []:

                attributes = vertex.get(
                    "attributes",
                    {},
                )

                if not is_fraudulent(
                    attributes.get(
                        "is_fraud"
                    )
                ):
                    continue

                fraud_records.append(
                    {
                        "transaction_id": vertex.get(
                            "v_id"
                        ),
                        "amount": attributes.get(
                            "amount"
                        ),
                        "is_fraud": attributes.get(
                            "is_fraud"
                        ),
                        "transaction_time": attributes.get(
                            "transaction_time"
                        ),
                        "merchant": merchant[
                            "merchant"
                        ],
                    }
                )

        fraud_records.sort(
            key=lambda x: float(
                x.get(
                    "amount",
                    0
                )
                or 0
            ),
            reverse=True,
        )

        total_count = len(
            fraud_records
        )

        print(
            "DEBUG merchant fraudulent transactions found:",
            total_count,
        )

        return {
            "records": fraud_records[
                :limit
            ],
            "total_count": total_count,
        }

    except Exception as e:

        print(
            "Merchant fraud transaction query error:",
            type(e).__name__,
            ":",
            e,
        )

        return {
            "records": [],
            "total_count": 0,
        }


# ============================================================
# NATURAL LANGUAGE SEARCH
# ============================================================

def search_fraud_question(question):

    if not question:

        return {
            "type": "unknown",
            "records": [],
            "count": 0,
            "answer": (
                "Please provide a fraud-related query."
            ),
        }

    q = str(
        question
    ).lower().strip()

    q = re.sub(
        r"\s+",
        " ",
        q,
    )

    # ========================================================
    # 1. SPECIFIC TRANSACTION
    # ========================================================

    match = re.search(
        r"\btransaction\s+(\d+)\b",
        q,
    )

    if match:

        transaction_id = match.group(1)

        # ----------------------------------------------------
        # SPECIAL / INVESTIGATION QUESTIONS
        # ----------------------------------------------------

        investigation_words = [
            "why",
            "suspicious",
            "investigate",
            "investigation",
            "explain",
            "reason",
            "evidence",
            "risk",
            "fraudulent",
        ]

        if any(
            word in q
            for word in investigation_words
        ):

            return investigate_transaction(
                transaction_id
            )

        # ----------------------------------------------------
        # CONNECTED TRANSACTION QUESTIONS
        # ----------------------------------------------------

        connected_words = [
            "connected",
            "associated",
            "linked",
            "related",
            "same card",
            "same merchant",
            "involving",
        ]

        if any(
            word in q
            for word in connected_words
        ):

            records = get_connected_transactions(
                transaction_id,
                limit=10,
            )

            return {
                "type": "connected_transactions",
                "transaction_id": transaction_id,
                "records": records,
                "count": len(records),
            }

        # ----------------------------------------------------
        # BASIC TRANSACTION LOOKUP
        # ----------------------------------------------------

        investigation = investigate_transaction(
    transaction_id
)

        return {
    "type": "transaction_investigation",
    "transaction_id": transaction_id,
    "transaction": investigation.get("transaction"),
    "merchant_id": investigation.get("merchant_id"),
    "card_id": investigation.get("card_id"),
    "evidence": investigation.get("evidence", []),
    "records": (
        [investigation.get("transaction")]
        if investigation.get("transaction")
        else []
    ),
    "count": (
        1
        if investigation.get("transaction")
        else 0
    ),
    "answer": explain_transaction_investigation(
        investigation
    ),
}

    # ========================================================
    # 2. HIGH VALUE FRAUD
    # ========================================================

    amount = extract_amount(q)

    if (
        amount is not None
        and any(
            word in q
            for word in [
                "fraud",
                "fraudulent",
                "suspicious",
            ]
        )
    ):

        print(
            "DEBUG extracted amount:",
            amount,
        )

        result = search_high_value_fraud(
            minimum_amount=amount,
            limit=20,
        )

        return {
            "type": "high_value_fraud",
            "minimum_amount": amount,
            "records": result[
                "records"
            ],
            "total_count": result[
                "total_count"
            ],
            "count": result[
                "total_count"
            ],
        }

    # ========================================================
    # 3. MERCHANT QUERY
    # ========================================================

    merchant_match = re.search(
        r"\b(?:show|find|get|list)\s+"
        r"(?:all\s+)?"
        r"(?:fraudulent\s+|fraud\s+|suspicious\s+)?"
        r"(?:transactions\s+)?"
        r"(?:for|from|of|at|involving|associated\s+with|linked\s+to)\s+"
        r"(?:merchant\s+)?"
        r"(.+?)"
        r"(?:\s+please)?$",
        q,
    )

    if merchant_match:

        merchant_name = (
            merchant_match
            .group(1)
            .strip()
        )

        merchant_name = merchant_name.rstrip(
            "?.!,"
        )

        print(
            "DEBUG merchant name from search:",
            repr(merchant_name),
        )

        merchant = get_merchant(
            merchant_name
        )

        print(
            "DEBUG merchant result:",
            merchant,
        )

        if merchant is None:

            return {
                "type": "merchant",
                "merchant": merchant_name,
                "records": [],
                "count": 0,
            }

        if any(
            word in q
            for word in [
                "fraud",
                "fraudulent",
                "suspicious",
            ]
        ):

            result = get_merchant_fraud_transactions(
                merchant_name,
                limit=20,
            )

            return {
                "type": "merchant_fraud",
                "merchant": merchant[
                    "merchant"
                ],
                "merchant_id": merchant[
                    "merchant_id"
                ],
                "records": result[
                    "records"
                ],
                "total_count": result[
                    "total_count"
                ],
                "count": result[
                    "total_count"
                ],
            }

        records = get_merchant_transactions(
            merchant_name,
            limit=20,
        )

        return {
            "type": "merchant",
            "merchant": merchant[
                "merchant"
            ],
            "merchant_id": merchant[
                "merchant_id"
            ],
            "records": records,
            "count": len(records),
        }

    # ========================================================
    # 4. TOP FRAUD
    # ========================================================

    if (
        any(
            word in q
            for word in [
                "highest",
                "largest",
                "top",
                "maximum",
                "most expensive",
            ]
        )
        and any(
            word in q
            for word in [
                "fraud",
                "fraudulent",
                "suspicious",
            ]
        )
    ):

        records = get_top_fraud_transactions(
            limit=10
        )

        return {
            "type": "top_fraud",
            "records": records,
            "count": len(records),
        }

    # ========================================================
    # 5. GENERAL FRAUD
    # ========================================================

    if any(
        word in q
        for word in [
            "fraud",
            "fraudulent",
            "suspicious",
        ]
    ):

        records = search_fraud_transactions(
            limit=20
        )

        return {
            "type": "fraud_transactions",
            "records": records,
            "count": len(records),
        }

    # ========================================================
    # 6. UNKNOWN
    # ========================================================

    return {
        "type": "unknown",
        "records": [],
        "count": 0,
        "answer": (
            "I could not determine "
            "the requested fraud query."
        ),
    }