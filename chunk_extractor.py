# Extracts short, concrete, embeddable chunks (skills/occupation candidates) from CV text.
# python -m spacy download en_core_web_lg

from __future__ import annotations

from typing import Iterable

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span

_NLP = None


def _get_nlp():
    """Load and cache the spaCy en_core_web_lg model."""
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_lg")
        except OSError as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_lg' is not installed. "
                "Run: python -m spacy download en_core_web_lg"
            ) from exc
    return _NLP


def _normalize_span_text(text: str) -> str:
    """Normalize chunk text for matching."""
    return " ".join(text.split()).strip().lower()


def _word_count(text: str) -> int:
    """Count words in a whitespace-normalized phrase."""
    if not text:
        return 0
    return len(text.split())


def _iter_verb_phrases(doc: Doc) -> Iterable[Span]:
    """Yield TOKEN VERB TOKEN spans using spaCy Matcher."""
    matcher = Matcher(doc.vocab)
    pattern = [
        {"IS_ALPHA": True},
        {"POS": "VERB"},
        {"IS_ALPHA": True},
    ]
    matcher.add("VERB_PHRASE", [pattern])
    for _, start, end in matcher(doc):
        yield doc[start:end]


def extract_chunks(text: str) -> list[str]:
    """Extract short candidate chunks from raw CV text.

    Sources:
    - noun chunks
    - named entities
    - TOKEN VERB TOKEN phrases via spaCy Matcher

    Filtering:
    - lowercase
    - keep 2 to 6 words (inclusive)
    - deduplicate while preserving first-seen order

    Args:
        text: Raw CV text.

    Returns:
        List of short chunk strings.
    """
    if not text or not text.strip():
        return []

    nlp = _get_nlp()
    doc = nlp(text)

    raw_chunks: list[str] = []
    raw_chunks.extend(span.text for span in doc.noun_chunks)
    raw_chunks.extend(ent.text for ent in doc.ents)
    raw_chunks.extend(span.text for span in _iter_verb_phrases(doc))

    seen: set[str] = set()
    final_chunks: list[str] = []
    for chunk in raw_chunks:
        normalized = _normalize_span_text(chunk)
        wc = _word_count(normalized)
        if wc < 2 or wc > 6:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        final_chunks.append(normalized)

    return final_chunks
