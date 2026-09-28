import json
import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_FILE = "data/corpus.jsonl"
EMBEDDINGS_FILE = "rag/embeddings.npy"

STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were",
    "who", "what", "when", "where", "which", "how",
    "did", "do", "does", "at", "in", "on", "of",
    "to", "for", "and", "or", "with", "by", "from",
    "before", "after", "won", "win"
}


def normalize_text(text):
    text = text.lower()

    # Make things like 1000m and 1000 metres easier to match
    text = re.sub(r"(\d+)\s*m\b", r"\1 metres", text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s-]", " ", text)

    return text


def get_keywords(text):
    text = normalize_text(text)

    words = text.split()

    return [
        word for word in words
        if word not in STOPWORDS
        and len(word) > 1
    ]


print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

# Load documents
documents = []

with open(CORPUS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            documents.append(json.loads(line))

print(f"Loaded {len(documents)} documents.")

# Load saved embeddings
if not os.path.exists(EMBEDDINGS_FILE):
    print("Embeddings file not found.")
    print("Run semantic_search.py once to create embeddings.")
    exit()

print("Loading saved embeddings...")

embeddings = np.load(EMBEDDINGS_FILE)

# User question
query = input("\nEnter your question: ")

print("\nSearching...")

# -----------------------------
# SEMANTIC SEARCH
# -----------------------------

query_embedding = model.encode([query])[0]

semantic_scores = []

for i, embedding in enumerate(embeddings):

    score = embedding @ query_embedding

    semantic_scores.append(score)


# -----------------------------
# KEYWORD SEARCH
# -----------------------------

query_keywords = get_keywords(query)

keyword_scores = []

for doc in documents:

    title = normalize_text(doc["title"])
    text = normalize_text(doc["text"])

    score = 0

    for keyword in query_keywords:

        # Title matches are more important
        if keyword in title.split():
            score += 5

        # Text matches
        score += text.count(keyword)

    keyword_scores.append(score)


# -----------------------------
# NORMALIZE KEYWORD SCORES
# -----------------------------

max_keyword = max(keyword_scores)

if max_keyword > 0:

    normalized_keywords = [
        score / max_keyword
        for score in keyword_scores
    ]

else:

    normalized_keywords = [
        0
        for _ in keyword_scores
    ]


# -----------------------------
# COMBINE SCORES
# -----------------------------

combined_scores = []

for i in range(len(documents)):

    semantic = semantic_scores[i]

    keyword = normalized_keywords[i]

    # Semantic search = 75%
    # Keyword search = 25%

    combined = (
        0.75 * semantic
        + 0.25 * keyword
    )

    combined_scores.append(
        (combined, i, semantic, keyword)
    )


# Highest score first
combined_scores.sort(
    reverse=True
)


# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("\n" + "=" * 70)
print("HYBRID SEARCH RESULTS")
print("=" * 70)

for rank, (
    combined,
    i,
    semantic,
    keyword
) in enumerate(combined_scores[:5], start=1):

    doc = documents[i]

    print(f"\n[{rank}]")
    print(f"Combined Score : {combined:.4f}")
    print(f"Semantic Score : {semantic:.4f}")
    print(f"Keyword Score  : {keyword:.4f}")

    print(f"Title          : {doc['title']}")
    print(f"Doc ID         : {doc['doc_id']}")
    print(f"URL            : {doc['url']}")

    print("\nEvidence:")
    print(doc["text"][:700])

print("\n" + "=" * 70)
print("HYBRID SEARCH COMPLETE")
print("=" * 70)