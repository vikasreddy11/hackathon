"""
graphrag/entity_extractor.py
============================
Extracts entity mentions from a user query using spaCy NER, noun chunk
extraction, and matching against known knowledge graph entity identifiers.
"""

from __future__ import annotations

import re
from typing import Iterable

import spacy
from spacy.language import Language


class EntityExtractor:
    """Extracts candidate entities from text queries."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        self.spacy_model = spacy_model
        self._nlp: Language | None = None

    def _get_nlp(self) -> Language:
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.spacy_model)
            except OSError:
                print(f"[EntityExtractor] Downloading spaCy model: {self.spacy_model}...")
                spacy.cli.download(self.spacy_model)
                self._nlp = spacy.load(self.spacy_model)
        return self._nlp

    def extract_entities(
        self,
        query: str,
        known_entities: Iterable[str] | None = None,
    ) -> list[str]:
        """
        Extract entity mentions and key concepts from query.

        Parameters
        ----------
        query : str
            The input question or text.
        known_entities : Iterable[str] | None
            Optional collection of known node_ids/labels from the Knowledge Graph
            to match against directly.

        Returns
        -------
        list[str]
            Deduplicated list of extracted entity phrases.
        """
        if not query or not query.strip():
            return []

        nlp = self._get_nlp()
        doc = nlp(query)

        candidates: list[str] = []

        # 1. Named Entities recognised by spaCy
        for ent in doc.ents:
            text = ent.text.strip().lower()
            if len(text) > 1 and not text.isnumeric():
                candidates.append(text)

        # 2. Noun chunks (e.g. 'knowledge graphs', 'retrieval systems', 'large language models')
        for chunk in doc.noun_chunks:
            # Strip leading determiners/pronouns like 'the', 'a', 'what', 'how'
            clean_chunk = re.sub(r"^(the|a|an|this|that|these|those|what|which|how|why)\s+", "", chunk.text.strip(), flags=re.IGNORECASE)
            clean_chunk = clean_chunk.strip().lower()
            if len(clean_chunk) > 2 and clean_chunk not in candidates:
                candidates.append(clean_chunk)

        # 3. Individual content tokens (proper nouns and technical nouns)
        for token in doc:
            if token.pos_ in ("PROPN", "NOUN") and not token.is_stop:
                token_lower = token.text.strip().lower()
                if len(token_lower) > 2 and token_lower not in candidates:
                    candidates.append(token_lower)

        # 4. Direct match against known graph entities if provided
        if known_entities:
            query_lower = query.lower()
            for ke in known_entities:
                ke_clean = ke.strip().lower()
                if not ke_clean or len(ke_clean) < 3:
                    continue
                # Word boundary match
                pattern = r"\b" + re.escape(ke_clean) + r"\b"
                if re.search(pattern, query_lower):
                    if ke_clean not in candidates:
                        candidates.append(ke_clean)

        # Deduplicate while preserving order
        seen = set()
        deduped: list[str] = []
        for c in candidates:
            c_norm = c.strip().lower()
            if c_norm and c_norm not in seen:
                seen.add(c_norm)
                deduped.append(c_norm)

        return deduped


_default_extractor = EntityExtractor()


def extract_entities(query: str, known_entities: Iterable[str] | None = None) -> list[str]:
    """Functional convenience wrapper for entity extraction."""
    return _default_extractor.extract_entities(query, known_entities=known_entities)
