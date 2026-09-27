
# ============================================================
# agents/tools.py
# ============================================================

from typing import Any, Dict

from graph.graph_search import get_graph_answer


# ============================================================
# GRAPH SEARCH TOOL
# ============================================================

def rag_search(question: str) -> Dict[str, Any]:

    print()
    print("========== RAG SEARCH TOOL ==========")
    print("Calling graph search...")
    print()

    result = get_graph_answer(question)

    return result


# ============================================================
# GRAPH STATS TOOL
# ============================================================

def graph_stats(question: str) -> Dict[str, Any]:

    print()
    print("========== GRAPH STATS TOOL ==========")
    print("Calling graph statistics...")
    print()

    result = get_graph_answer(question)

    return result


# ============================================================
# MULTI-HOP TOOL
# ============================================================

def multi_hop_search(question: str) -> Dict[str, Any]:

    print()
    print("========== MULTI-HOP TOOL ==========")
    print("Calling multi-hop reasoning...")
    print()

    q = question.lower()

    # --------------------------------------------------------
    # COMPARE MOST ↔ FEWEST
    # --------------------------------------------------------

    if (
        "most" in q
        and "fewest" in q
    ):

        # ----------------------------------------------------
        # Determine the field
        # ----------------------------------------------------

        if (
            "nation" in q
            or "nations" in q
        ):

            field = "nations"

        else:

            field = "competitors"

        print(
            f"Multi-hop field: {field}"
        )

        print()

        # ----------------------------------------------------
        # HOP 1
        # ----------------------------------------------------

        print(
            f"========== HOP 1: "
            f"FIND MAXIMUM {field.upper()} =========="
        )

        print()

        max_question = (
            f"Which event had the most "
            f"{field} in 2018?"
        )

        max_result = rag_search(
            max_question
        )

        # ----------------------------------------------------
        # HOP 2
        # ----------------------------------------------------

        print(
            f"========== HOP 2: "
            f"FIND MINIMUM {field.upper()} =========="
        )

        print()

        min_question = (
            f"Which event had the fewest "
            f"{field} in 2018?"
        )

        min_result = rag_search(
            min_question
        )

        # ----------------------------------------------------
        # VALIDATION
        # ----------------------------------------------------

        print()
        print(
            "========== MULTI-HOP VALIDATION =========="
        )
        print()

        max_record = (
            max_result.get("record")
            if max_result
            else None
        )

        min_record = (
            min_result.get("record")
            if min_result
            else None
        )

        if max_record:

            print(
                f"✓ Maximum {field} record found"
            )

        else:

            print(
                f"✗ Maximum {field} record missing"
            )

        if min_record:

            print(
                f"✓ Minimum {field} record found"
            )

        else:

            print(
                f"✗ Minimum {field} record missing"
            )

        # ----------------------------------------------------
        # NUMERIC VALUES
        # ----------------------------------------------------

        max_value = (
            max_result.get("value")
            if max_result
            else None
        )

        min_value = (
            min_result.get("value")
            if min_result
            else None
        )

        difference = None

        if (
            isinstance(max_value, (int, float))
            and isinstance(min_value, (int, float))
        ):

            difference = (
                max_value - min_value
            )

        # ----------------------------------------------------
        # RETURN STRUCTURED EVIDENCE
        # ----------------------------------------------------

        return {
            "type": "multi_hop",

            "field": field,

            "max_result": max_result,

            "min_result": min_result,

            "max_record": max_record,

            "min_record": min_record,

            "max_value": max_value,

            "min_value": min_value,

            "difference": difference,

            "hops": 2,

            "answer": None,
        }

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    print(
        "Multi-hop pattern not recognized."
    )

    return {
        "type": "multi_hop",

        "answer": (
            "I could not determine the "
            "required multi-hop reasoning."
        ),

        "records": [],

        "hops": 0,
    }


# ============================================================
# TOOL REGISTRY
# ============================================================

TOOLS = {

    "graph_search": rag_search,

    "graph_stats": graph_stats,

    "multi_hop": multi_hop_search,
}


# ============================================================
# TOOL EXECUTOR
# ============================================================

def execute_tool(
    tool_name: str,
    question: str
) -> Dict[str, Any]:

    print()
    print(
        "========== TOOL EXECUTOR =========="
    )

    print(
        "Selected tool:",
        tool_name
    )

    print()

    tool = TOOLS.get(
        tool_name
    )

    # --------------------------------------------------------
    # UNKNOWN TOOL
    # --------------------------------------------------------

    if tool is None:

        print(
            f"Unknown tool: {tool_name}"
        )

        return {

            "type": "error",

            "answer": (
                f"Unknown tool: {tool_name}"
            ),

            "records": [],
        }

    # --------------------------------------------------------
    # EXECUTE
    # --------------------------------------------------------

    return tool(
        question
    )


# ============================================================
# TOOL LIST
# ============================================================

def available_tools():

    return list(
        TOOLS.keys()
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print(
        "Available tools:"
    )

    for tool in available_tools():

        print(
            f"- {tool}"
        )

    print()

    test_question = (
        "Compare the event with the most "
        "competitors to the event with the "
        "fewest competitors in 2018."
    )

    result = execute_tool(
        "multi_hop",
        test_question
    )

    print()
    print(
        "========== TOOL RESULT =========="
    )

    print(
        result
    )
