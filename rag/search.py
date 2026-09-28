import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CORPUS_FILE = "data/corpus.jsonl"
EMBEDDINGS_FILE = "rag/embeddings.npy"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "who",
    "what",
    "when",
    "where",
    "which",
    "how",
    "did",
    "do",
    "does",
    "at",
    "in",
    "on",
    "of",
    "to",
    "for",
    "and",
    "or",
    "with",
    "by",
    "from",
    "before",
    "after",
    "won",
    "win",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text):
    """
    Normalize text for keyword matching.
    """

    text = text.lower()

    # Convert:
    # 100m -> 100 metres
    text = re.sub(
        r"(\d+)\s*m\b",
        r"\1 metres",
        text
    )

    # Remove punctuation
    text = re.sub(
        r"[^a-z0-9\s-]",
        " ",
        text
    )

    # Normalize multiple spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def get_keywords(text):
    """
    Extract useful keywords from a question.
    """

    words = normalize_text(text).split()

    keywords = []

    for word in words:

        if word in STOPWORDS:
            continue

        if len(word) <= 1:
            continue

        keywords.append(word)

    return keywords


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading RAG embedding model...")

model = SentenceTransformer(
    MODEL_NAME
)


# ============================================================
# LOAD CORPUS
# ============================================================

documents = []

with open(
    CORPUS_FILE,
    "r",
    encoding="utf-8"
) as f:

    for line in f:

        if line.strip():

            documents.append(
                json.loads(line)
            )


print(
    f"Loaded {len(documents)} documents."
)


# ============================================================
# LOAD EMBEDDINGS
# ============================================================

if not os.path.exists(
    EMBEDDINGS_FILE
):

    raise FileNotFoundError(
        "rag/embeddings.npy not found. "
        "Run the embedding generation script first."
    )


embeddings = np.load(
    EMBEDDINGS_FILE
)


print("Embeddings loaded.")


# ============================================================
# SEARCH
# ============================================================

def search(query, top_k=5):
    """
    Hybrid RAG retrieval.

    Combines:
    1. Semantic similarity
    2. Title keyword matching
    3. Document keyword matching
    """

    # --------------------------------------------------------
    # Query embedding
    # --------------------------------------------------------

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )[0]


    # --------------------------------------------------------
    # Semantic similarity
    # --------------------------------------------------------

    semantic_scores = (
        embeddings @ query_embedding
    )


    # --------------------------------------------------------
    # Query keywords
    # --------------------------------------------------------

    query_keywords = get_keywords(
        query
    )


    # --------------------------------------------------------
    # Keyword scoring
    # --------------------------------------------------------

    keyword_scores = []


    for doc in documents:

        title = normalize_text(
            doc["title"]
        )

        text = normalize_text(
            doc["text"]
        )


        title_words = set(
            title.split()
        )


        score = 0


        for keyword in query_keywords:

            # Strong signal:
            # keyword appears in title
            if keyword in title_words:

                score += 10


            # Weak signal:
            # keyword appears anywhere
            # in document
            if keyword in text:

                score += 1


        keyword_scores.append(
            score
        )


    # --------------------------------------------------------
    # Normalize keyword scores
    # --------------------------------------------------------

    max_keyword = max(
        keyword_scores
    )


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


    # --------------------------------------------------------
    # Combine scores
    # --------------------------------------------------------

    results = []


    for i in range(
        len(documents)
    ):

        semantic_score = float(
            semantic_scores[i]
        )

        keyword_score = float(
            normalized_keywords[i]
        )


        # Semantic retrieval remains
        # the main signal.
        #
        # Keyword matching helps when
        # the question contains exact
        # event/entity names.

        combined_score = (
            0.75 * semantic_score
            + 0.25 * keyword_score
        )


        results.append({

            "score": combined_score,

            "semantic_score":
                semantic_score,

            "keyword_score":
                keyword_score,

            "document":
                documents[i]

        })


    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    return results[:top_k]


# ============================================================
# COMMAND LINE TEST
# ============================================================

if __name__ == "__main__":

    question = input(
        "\nEnter your question: "
    )


    results = search(
        question,
        top_k=5
    )


    print(
        "\nResults found:",
        len(results)
    )


    for rank, result in enumerate(
        results,
        start=1
    ):

        doc = result["document"]


        print(
            f"\n{rank}. {doc['title']}"
        )


        print(
            "Doc ID:",
            doc["doc_id"]
        )


        print(
            "Semantic score:",
            round(
                result["semantic_score"],
                4
            )
        )


        print(
            "Keyword score:",
            round(
                result["keyword_score"],
                4
            )
        )


        print(
            "Combined score:",
            round(
                result["score"],
                4
            )
        )