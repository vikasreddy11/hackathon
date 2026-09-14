"""
knowledge_graph/extractor.py
============================
Entity and relationship extraction layer.

Architecture
------------
BaseExtractor   — abstract interface (extract a single chunk → entities + relations)
SpacyExtractor  — default implementation using spaCy en_core_web_sm
LLMExtractor    — stub class ready for future implementation

Usage
-----
    from knowledge_graph.extractor import SpacyExtractor
    extractor = SpacyExtractor()
    entities, relations = extractor.extract("OpenAI was founded in San Francisco by Sam Altman.")
"""

from __future__ import annotations

import abc
import itertools
from dataclasses import dataclass
from typing import Any

from knowledge_graph.schema import EdgeType, EntityLabel


# ---------------------------------------------------------------------------
# Extracted entity / relation types (lightweight, extractor-level)
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    """A single named entity extracted from a text span."""
    text: str           # Raw entity surface form (e.g. "OpenAI")
    label: str          # NER label (e.g. "ORG")
    start_char: int     # Character offset within the chunk text
    end_char: int       # Character offset within the chunk text
    normalized: str = ""  # Lowercased, stripped form used as node_id

    def __post_init__(self):
        if not self.normalized:
            self.normalized = self.text.strip().lower()


@dataclass
class ExtractedRelation:
    """A directed relationship between two entities."""
    source_text: str    # Normalized source entity text
    target_text: str    # Normalized target entity text
    edge_type: EdgeType
    weight: float = 1.0
    metadata: dict[str, Any] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseExtractor(abc.ABC):
    """
    Abstract entity/relationship extractor.

    Subclasses implement ``extract``. The caller (builder.py) is agnostic to
    the underlying extraction technology, so spaCy and LLM backends are
    interchangeable at construction time.
    """

    @abc.abstractmethod
    def extract(
        self, text: str
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """
        Extract entities and relations from a text string.

        Parameters
        ----------
        text : str
            The chunk text to analyse.

        Returns
        -------
        entities : list[ExtractedEntity]
        relations : list[ExtractedRelation]
            Intra-chunk co-occurrence relations between entity pairs.
        """
        ...


# ---------------------------------------------------------------------------
# spaCy extractor (default)
# ---------------------------------------------------------------------------

# NER labels we care about — filters out CARDINAL, ORDINAL, PERCENT, etc.
_USEFUL_LABELS = {
    EntityLabel.PERSON.value,
    EntityLabel.ORG.value,
    EntityLabel.GPE.value,
    EntityLabel.PRODUCT.value,
    EntityLabel.EVENT.value,
    EntityLabel.WORK_OF_ART.value,
    EntityLabel.LAW.value,
    EntityLabel.LANGUAGE.value,
    EntityLabel.NORP.value,
    EntityLabel.FAC.value,
    EntityLabel.LOC.value,
}


class SpacyExtractor(BaseExtractor):
    """
    Entity extractor backed by spaCy's en_core_web_sm model.

    - Runs NER on the chunk text.
    - Filters to ``_USEFUL_LABELS`` to avoid noise.
    - Deduplicates entities by normalised text within a chunk.
    - Produces COOCCURS_WITH edges for all unique entity pairs in the chunk.

    Install requirements:
        pip install spacy
        python -m spacy download en_core_web_sm
    """

    def __init__(self, model_name: str = "en_core_web_sm"):
        try:
            import spacy  # noqa: PLC0415
            self._nlp = spacy.load(model_name)
        except OSError:
            raise RuntimeError(
                f"spaCy model '{model_name}' not found. "
                f"Run: python -m spacy download {model_name}"
            ) from None
        except ImportError:
            raise RuntimeError(
                "spaCy is not installed. Run: pip install spacy"
            ) from None

    def extract(
        self, text: str
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        doc = self._nlp(text)

        # --- Entities --------------------------------------------------------
        seen_normalized: set[str] = set()
        entities: list[ExtractedEntity] = []

        for ent in doc.ents:
            if ent.label_ not in _USEFUL_LABELS:
                continue
            norm = ent.text.strip().lower()
            if norm in seen_normalized or not norm:
                continue
            seen_normalized.add(norm)
            entities.append(
                ExtractedEntity(
                    text=ent.text.strip(),
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    normalized=norm,
                )
            )

        # --- Relations (co-occurrence within chunk) --------------------------
        relations: list[ExtractedRelation] = []
        for a, b in itertools.combinations(entities, 2):
            relations.append(
                ExtractedRelation(
                    source_text=a.normalized,
                    target_text=b.normalized,
                    edge_type=EdgeType.COOCCURS_WITH,
                    weight=1.0,
                )
            )

        return entities, relations


# ---------------------------------------------------------------------------
# LLM extractor stub (to be implemented by future team members)
# ---------------------------------------------------------------------------

class LLMExtractor(BaseExtractor):
    """
    Placeholder LLM-based extractor — swap in to replace SpacyExtractor.

    Expected implementation:
        - Send chunk text to an LLM with a structured prompt requesting
          entity/relation JSON output.
        - Parse the response into ExtractedEntity / ExtractedRelation objects.
        - Return the same (entities, relations) tuple as SpacyExtractor.

    The builder.py is fully agnostic to which extractor is in use.
    """

    def __init__(self, model: str = "gpt-4o-mini", **kwargs: Any):
        self.model = model
        self.kwargs = kwargs
        # TODO: initialise your LLM client here (e.g. openai.OpenAI())
        raise NotImplementedError(
            "LLMExtractor is not yet implemented. "
            "Use SpacyExtractor or implement this class."
        )

    def extract(
        self, text: str
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        # TODO: implement prompt + parse loop
        raise NotImplementedError
