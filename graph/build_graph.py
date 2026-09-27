import json
import re

CORPUS_FILE = "data/corpus.jsonl"
OUTPUT_FILE = "graph/graph_data.jsonl"


def extract_infobox(text):
    data = {}

    for line in text.splitlines():
        line = line.strip()

        if ":" not in line:
            continue

        key, value = line.split(":", 1)

        key = key.strip()
        value = value.strip()

        if key in {
            "event",
            "games",
            "venue",
            "date",
            "competitors",
            "nations",
            "gold",
            "silver",
            "bronze",
        }:
            data[key] = value

    return data


def clean_team_medalist(value, text, medal):
    """
    The corpus sometimes stores team medalists without separators,
    for example:

        Dani KingLaura TrottJoanna Rowsell

    The article body contains the correctly separated names.
    Extract the team members from the article text when possible.
    """

    if not value:
        return value

    # If the value already contains obvious separators, keep it.
    if "," in value or " and " in value:
        return value

    # Find a sentence containing the medal result.
    if medal == "gold":
        patterns = [
            r"The Great Britain team consisting of (.*?) won the gold medal",
            r"consisting of (.*?) won the gold medal",
        ]
    elif medal == "silver":
        patterns = [
            r"(.*?) took the silver medal",
            r"(.*?) won the silver medal",
        ]
    elif medal == "bronze":
        patterns = [
            r"(.*?) won bronze",
            r"(.*?) won the bronze medal",
        ]
    else:
        patterns = []

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)

        if match:
            extracted = match.group(1).strip()

            # Remove unnecessary leading text.
            extracted = re.sub(
                r"^(?:the\s+)?(?:Great Britain|United States|Canada)\s+team\s+",
                "",
                extracted,
                flags=re.IGNORECASE,
            )

            # Normalize "A, B and C".
            extracted = re.sub(
                r"\s+and\s+",
                ", ",
                extracted,
                flags=re.IGNORECASE,
            )

            return extracted

    return value


print("Loading corpus...")

documents = []

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            documents.append(json.loads(line))

print(f"Loaded {len(documents)} documents.")


graph_records = []

for doc in documents:

    text = doc["text"]

    info = extract_infobox(text)

    gold = clean_team_medalist(
        info.get("gold"),
        text,
        "gold",
    )

    silver = clean_team_medalist(
        info.get("silver"),
        text,
        "silver",
    )

    bronze = clean_team_medalist(
        info.get("bronze"),
        text,
        "bronze",
    )

    record = {
        "doc_id": doc["doc_id"],
        "title": doc["title"],
        "event": info.get("event"),
        "games": info.get("games"),
        "venue": info.get("venue"),
        "date": info.get("date"),
        "competitors": info.get("competitors"),
        "nations": info.get("nations"),
        "gold": gold,
        "silver": silver,
        "bronze": bronze,
    }

    graph_records.append(record)


with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for record in graph_records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            + "\n"
        )


print(f"Graph data saved to: {OUTPUT_FILE}")
print(f"Created {len(graph_records)} graph records.")