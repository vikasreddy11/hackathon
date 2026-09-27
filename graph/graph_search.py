# ============================================================
# graph/graph_search.py
# ============================================================

import json
import os
import re
from typing import Any, Dict, List, Optional


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

GRAPH_DATA_FILE = os.path.join(
    BASE_DIR,
    "graph",
    "graph_data.jsonl"
)


# ============================================================
# LOAD GRAPH DATA
# ============================================================

def load_graph_records() -> List[Dict[str, Any]]:

    records = []

    if not os.path.exists(GRAPH_DATA_FILE):

        print(
            f"ERROR: Graph data file not found: "
            f"{GRAPH_DATA_FILE}"
        )

        return records

    with open(
        GRAPH_DATA_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                records.append(
                    json.loads(line)
                )

            except json.JSONDecodeError:
                continue

    return records


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(text: Any) -> str:

    if text is None:
        return ""

    text = str(text).lower()

    replacements = {
        "’": "'",
        "‘": "'",
        "–": "-",
        "—": "-",
        "−": "-",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(
        r"[^\w\s'-]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# NUMBER EXTRACTION
# ============================================================

def extract_number(value: Any) -> Optional[int]:

    if value is None:
        return None

    match = re.search(
        r"\d+",
        str(value)
    )

    if not match:
        return None

    try:
        return int(
            match.group()
        )

    except ValueError:
        return None


# ============================================================
# YEAR EXTRACTION
# ============================================================

def extract_year(
    question: str
) -> Optional[int]:

    match = re.search(
        r"\b(19|20)\d{2}\b",
        question
    )

    if not match:
        return None

    return int(
        match.group()
    )


# ============================================================
# QUESTION TYPE
# ============================================================

def detect_question_type(
    question: str
) -> str:

    q = normalize(question)

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    comparison_patterns = [
        r"\bcompare\b",
        r"\bcomparison\b",
        r"\bdifference between\b",
        r"\bversus\b",
        r"\bvs\b",
        r"\bsimilar\b",
        r"\bdifferent\b",
    ]

    for pattern in comparison_patterns:

        if re.search(
            pattern,
            q
        ):
            return "comparison"

    # --------------------------------------------------------
    # Superlative
    # --------------------------------------------------------

    superlative_patterns = [
        r"\bfewest\b",
        r"\bmost\b",
        r"\blargest\b",
        r"\bsmallest\b",
        r"\bhighest\b",
        r"\blowest\b",
        r"\bmaximum\b",
        r"\bminimum\b",
    ]

    for pattern in superlative_patterns:

        if re.search(
            pattern,
            q
        ):
            return "superlative"

    # --------------------------------------------------------
    # Direct count lookup
    # --------------------------------------------------------

    direct_count_patterns = [
        r"\bhow many nations\b",
        r"\bhow many competitors\b",
        r"\bhow many athletes\b",
        r"\bnumber of nations\b",
        r"\bnumber of competitors\b",
        r"\bnumber of athletes\b",
    ]

    for pattern in direct_count_patterns:

        if re.search(
            pattern,
            q
        ):
            return "lookup"

    # --------------------------------------------------------
    # Aggregation
    # --------------------------------------------------------

    aggregation_patterns = [
        r"\bwhich events?\b",
        r"\bhow many events?\b",
        r"\bmore than\b",
        r"\bgreater than\b",
        r"\bless than\b",
        r"\bfewer than\b",
        r"\bat least\b",
        r"\bat most\b",
        r"\bexactly\b",
    ]

    for pattern in aggregation_patterns:

        if re.search(
            pattern,
            q
        ):
            return "aggregation"

    # --------------------------------------------------------
    # Explanation
    # --------------------------------------------------------

    explanation_patterns = [
        r"\bexplain\b",
        r"\bwhy\b",
        r"\bsignificance\b",
        r"\bimportance\b",
        r"\bdescribe\b",
        r"\bdiscuss\b",
        r"\btell me about\b",
        r"\bhow was\b",
        r"\bhow did\b",
    ]

    for pattern in explanation_patterns:

        if re.search(
            pattern,
            q
        ):
            return "explanation"

    return "lookup"


# ============================================================
# FIELD DETECTION
# ============================================================

def detect_field(
    question: str
) -> Optional[str]:

    q = normalize(question)

    if (
        "nation" in q
        or "nations" in q
    ):
        return "nations"

    if (
        "competitor" in q
        or "competitors" in q
    ):
        return "competitors"

    if (
        "athlete" in q
        or "athletes" in q
    ):
        return "competitors"

    if "gold" in q:
        return "gold"

    if "silver" in q:
        return "silver"

    if "bronze" in q:
        return "bronze"

    if "venue" in q:
        return "venue"

    if "date" in q:
        return "date"

    return None


# ============================================================
# EVENT KEYWORDS
# ============================================================

def extract_event_keywords(
    question: str
) -> List[str]:

    q = normalize(question)

    event_terms = [
        "sprint",
        "individual",
        "pursuit",
        "mass start",
        "relay",
        "mixed relay",
        "singles",
        "pair skating",
        "ice dance",
        "ski jumping",
        "downhill",
        "slalom",
        "giant slalom",
        "snowboard",
        "freestyle",
        "speed skating",
        "figure skating",
        "biathlon",
    ]

    keywords = []

    for term in event_terms:

        if term in q:
            keywords.append(term)

    return keywords


# ============================================================
# FILTER BY YEAR
# ============================================================

def filter_year(
    records: List[Dict[str, Any]],
    year: Optional[int]
) -> List[Dict[str, Any]]:

    if year is None:
        return records

    result = []

    for record in records:

        games = str(
            record.get("games") or ""
        )

        if str(year) in games:

            result.append(record)

    return result


# ============================================================
# SCORE RECORD
# ============================================================

def score_record(
    question: str,
    record: Dict[str, Any]
) -> int:

    q = normalize(question)

    score = 0

    title = normalize(
        record.get("title")
    )

    event = normalize(
        record.get("event")
    )

    games = normalize(
        record.get("games")
    )

    q_tokens = set(
        q.split()
    )

    title_tokens = set(
        title.split()
    )

    event_tokens = set(
        event.split()
    )

    # General token matching

    score += (
        len(
            q_tokens & title_tokens
        ) * 3
    )

    score += (
        len(
            q_tokens & event_tokens
        ) * 5
    )

    # Year

    year = extract_year(
        question
    )

    if year is not None:

        if str(year) in games:

            score += 20

    # Winter Olympics

    if "winter olympics" in q:

        if "winter" in games:

            score += 10

    # Event keywords

    for keyword in extract_event_keywords(
        question
    ):

        if keyword in title:
            score += 15

        if keyword in event:
            score += 20

    # Biathlon

    if "biathlon" in q:

        if "biathlon" in title:
            score += 15

    return score


# ============================================================
# RELEVANT RECORDS
# ============================================================

def find_relevant_records(
    question: str,
    records: List[Dict[str, Any]],
    limit: int = 50
) -> List[Dict[str, Any]]:

    year = extract_year(
        question
    )

    year_records = filter_year(
        records,
        year
    )

    scored = []

    for record in year_records:

        score = score_record(
            question,
            record
        )

        if score > 0:

            scored.append(
                (
                    score,
                    record
                )
            )

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        record
        for score, record
        in scored[:limit]
    ]


# ============================================================
# LOOKUP
# ============================================================

def lookup(
    question: str,
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:

    relevant = find_relevant_records(
        question,
        records,
        limit=10
    )

    if not relevant:

        return {
            "type": "lookup",
            "answer": None,
            "records": []
        }

    record = relevant[0]

    field = detect_field(
        question
    )

    if field:

        answer = record.get(
            field
        )

    else:

        answer = record.get(
            "title"
        )

    return {
        "type": "lookup",
        "answer": answer,
        "records": relevant,
        **record
    }


# ============================================================
# CONDITIONS
# ============================================================

def detect_conditions(
    question: str
) -> List[Dict[str, Any]]:

    q = normalize(question)

    conditions = []

    # --------------------------------------------------------
    # Competitors
    # --------------------------------------------------------

    competitor_patterns = [

        (
            r"\b(?:more than|greater than)\s+(\d+)\s+competitors?\b",
            ">"
        ),

        (
            r"\b(?:less than|fewer than)\s+(\d+)\s+competitors?\b",
            "<"
        ),

        (
            r"\bat least\s+(\d+)\s+competitors?\b",
            ">="
        ),

        (
            r"\bat most\s+(\d+)\s+competitors?\b",
            "<="
        ),

        (
            r"\bexactly\s+(\d+)\s+competitors?\b",
            "=="
        ),
    ]

    for pattern, operator in competitor_patterns:

        matches = re.findall(
            pattern,
            q
        )

        for match in matches:

            conditions.append(
                {
                    "field": "competitors",
                    "operator": operator,
                    "value": int(match)
                }
            )

    # --------------------------------------------------------
    # Nations
    # --------------------------------------------------------

    nation_patterns = [

        (
            r"\b(?:more than|greater than)\s+(\d+)\s+nations?\b",
            ">"
        ),

        (
            r"\b(?:less than|fewer than)\s+(\d+)\s+nations?\b",
            "<"
        ),

        (
            r"\bat least\s+(\d+)\s+nations?\b",
            ">="
        ),

        (
            r"\bat most\s+(\d+)\s+nations?\b",
            "<="
        ),

        (
            r"\bexactly\s+(\d+)\s+nations?\b",
            "=="
        ),
    ]

    for pattern, operator in nation_patterns:

        matches = re.findall(
            pattern,
            q
        )

        for match in matches:

            conditions.append(
                {
                    "field": "nations",
                    "operator": operator,
                    "value": int(match)
                }
            )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    unique = []

    seen = set()

    for condition in conditions:

        key = (
            condition["field"],
            condition["operator"],
            condition["value"]
        )

        if key not in seen:

            seen.add(key)

            unique.append(
                condition
            )

    return unique


# ============================================================
# APPLY CONDITIONS
# ============================================================

def apply_conditions(
    records: List[Dict[str, Any]],
    conditions: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:

    result = []

    for record in records:

        valid = True

        for condition in conditions:

            field = condition["field"]

            operator = condition["operator"]

            target = condition["value"]

            actual = extract_number(
                record.get(field)
            )

            if actual is None:

                valid = False
                break

            if operator == ">":

                if not actual > target:
                    valid = False
                    break

            elif operator == "<":

                if not actual < target:
                    valid = False
                    break

            elif operator == ">=":

                if not actual >= target:
                    valid = False
                    break

            elif operator == "<=":

                if not actual <= target:
                    valid = False
                    break

            elif operator == "==":

                if not actual == target:
                    valid = False
                    break

        if valid:

            result.append(record)

    return result


# ============================================================
# AGGREGATION
# ============================================================

def aggregation(
    question: str,
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:

    year = extract_year(
        question
    )

    year_records = filter_year(
        records,
        year
    )

    conditions = detect_conditions(
        question
    )

    print(
        "Detected conditions:",
        conditions
    )

    if conditions:

        matched = apply_conditions(
            year_records,
            conditions
        )

    else:

        matched = find_relevant_records(
            question,
            year_records,
            limit=50
        )

    return {
        "type": "aggregation",
        "answer": len(matched),
        "count": len(matched),
        "records": matched
    }


# ============================================================
# SUPERLATIVE
# ============================================================


def superlative(
    question: str,
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:

    year = extract_year(question)

    year_records = filter_year(
        records,
        year
    )

    if not year_records:

        return {
            "type": "superlative",
            "answer": None,
            "record": None,
            "records": []
        }

    q = normalize(question)

    # --------------------------------------------------------
    # Determine field
    # --------------------------------------------------------

    if "nation" in q:

        field = "nations"

    else:

        field = "competitors"

    numeric_records = []

    for record in year_records:

        value = extract_number(
            record.get(field)
        )

        if value is not None:

            numeric_records.append(
                (value, record)
            )

    if not numeric_records:

        return {
            "type": "superlative",
            "answer": None,
            "record": None,
            "records": []
        }

    # --------------------------------------------------------
    # Determine MINIMUM / MAXIMUM
    # --------------------------------------------------------

    is_minimum = any(
        word in q
        for word in [
            "fewest",
            "smallest",
            "minimum",
            "lowest"
        ]
    )

    if is_minimum:

        target_value = min(
            value
            for value, record
            in numeric_records
        )

    else:

        target_value = max(
            value
            for value, record
            in numeric_records
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Keep ALL records tied at the target value.
    # --------------------------------------------------------

    tied_records = [
        record
        for value, record
        in numeric_records
        if value == target_value
    ]

    # --------------------------------------------------------
    # Remove accidental duplicates.
    # --------------------------------------------------------

    unique_records = []

    seen = set()

    for record in tied_records:

        key = (
            record.get("title"),
            record.get("event"),
            record.get("games"),
            record.get(field)
        )

        if key not in seen:

            seen.add(key)

            unique_records.append(
                record
            )

    # --------------------------------------------------------
    # Backward-compatible primary record.
    # --------------------------------------------------------

    primary_record = (
        unique_records[0]
        if unique_records
        else None
    )

    return {
        "type": "superlative",

        "answer": (
            primary_record.get("title")
            if primary_record
            else None
        ),

        "value": target_value,

        "field": field,

        "record": primary_record,

        "records": unique_records,

        "count": len(unique_records),

        **(
            primary_record
            if primary_record
            else {}
        )
    }



# ============================================================
# EXPLANATION
# ============================================================

def explanation(
    question: str,
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:

    relevant = find_relevant_records(
        question,
        records,
        limit=5
    )

    if not relevant:

        return {
            "type": "explanation",
            "answer": (
                "I could not find enough "
                "relevant information in the graph."
            ),
            "records": []
        }

    record = relevant[0]

    title = record.get(
        "title"
    )

    event = record.get(
        "event"
    )

    games = record.get(
        "games"
    )

    venue = record.get(
        "venue"
    )

    date = record.get(
        "date"
    )

    competitors = record.get(
        "competitors"
    )

    nations = record.get(
        "nations"
    )

    gold = record.get(
        "gold"
    )

    silver = record.get(
        "silver"
    )

    bronze = record.get(
        "bronze"
    )

    answer = (
        f"{title} was an event at the {games}. "
        f"The event was the {event}"
    )

    if date:
        answer += f", held on {date}"

    if venue:
        answer += f" at {venue}"

    answer += "."

    if competitors:

        answer += (
            f" It involved {competitors} competitors"
        )

    if nations:

        answer += (
            f" from {nations} nations"
        )

    answer += "."

    if gold:

        answer += (
            f" The gold medal was won by {gold}."
        )

    if silver:

        answer += (
            f" Silver went to {silver}."
        )

    if bronze:

        answer += (
            f" Bronze went to {bronze}."
        )

    return {
        "type": "explanation",
        "answer": answer,
        "records": relevant
    }


# ============================================================
# COMPARISON
# ============================================================

def comparison(
    question: str,
    records: List[Dict[str, Any]]
) -> Dict[str, Any]:

    relevant = find_relevant_records(
        question,
        records,
        limit=20
    )

    # --------------------------------------------------------
    # Explicit event extraction
    # --------------------------------------------------------

    q = normalize(
        question
    )

    event_keywords = extract_event_keywords(
        question
    )

    selected = []

    for record in relevant:

        title = normalize(
            record.get("title")
        )

        event = normalize(
            record.get("event")
        )

        matches = 0

        for keyword in event_keywords:

            if keyword in title:
                matches += 1

            if keyword in event:
                matches += 1

        if matches > 0:

            selected.append(
                record
            )

    # Remove duplicates

    unique = []

    seen_titles = set()

    for record in selected:

        title = record.get(
            "title"
        )

        if title not in seen_titles:

            seen_titles.add(
                title
            )

            unique.append(
                record
            )

    if len(unique) >= 2:

        relevant = unique[:2]

    if len(relevant) < 2:

        return {
            "type": "comparison",
            "answer": (
                "I could not find enough "
                "records to compare."
            ),
            "records": relevant
        }

    first = relevant[0]

    second = relevant[1]

    answer = (
        f"Comparison of "
        f"{first.get('event', first.get('title'))} "
        f"and "
        f"{second.get('event', second.get('title'))}:"
    )

    return {
        "type": "comparison",
        "answer": answer,
        "records": relevant
    }


# ============================================================
# MAIN
# ============================================================

def get_graph_answer(
    question: str
) -> Dict[str, Any]:

    print(
        "\n========== GRAPH SEARCH =========="
    )

    print(
        "Question:",
        question
    )

    records = load_graph_records()

    print(
        "Loaded graph records:",
        len(records)
    )

    question_type = detect_question_type(
        question
    )

    print(
        "Graph question type:",
        question_type
    )

    if question_type == "explanation":

        result = explanation(
            question,
            records
        )

    elif question_type == "comparison":

        result = comparison(
            question,
            records
        )

    elif question_type == "aggregation":

        result = aggregation(
            question,
            records
        )

    elif question_type == "superlative":

        result = superlative(
            question,
            records
        )

    else:

        result = lookup(
            question,
            records
        )

    return result


# ============================================================
# COMPATIBILITY ALIAS
# ============================================================

def search_graph(
    question: str
):

    return get_graph_answer(
        question
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    questions = [

        "Which event had the most competitors in 2018?",

        "Which event had the fewest competitors in 2018?",

        "Which event had the most nations in 2018?",

        "Which event had the fewest nations in 2018?",

        "Which event had exactly 80 competitors in 2018?",

        "Which events had fewer than 30 competitors in 2018?",

        "Compare the women's sprint and women's individual in 2018.",

        "Who won gold in the 2018 Winter Olympics womens sprint?",

    ]

    for question in questions:

        print(
            "\n" + "=" * 70
        )

        result = get_graph_answer(
            question
        )

        print(
            "\nRESULT:"
        )

        print(
            result
        )