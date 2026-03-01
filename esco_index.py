# Builds and caches ESCO occupation/skill embedding index from ESCO v1.1 CSV files.

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

OCCUPATIONS_CSV = "data/occupations_en.csv"
SKILLS_CSV = "data/skills_en.csv"
CACHE_PATH = "data/esco_embeddings.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def _clean_text(value: Any) -> str:
    """Return normalized string value from CSV cell."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() == "nan":
        return ""
    return text


def _build_embedding_text(
    preferred_label: str,
    alt_labels: str,
    description: str,
) -> str:
    """Build rich embedding text from preferred label, alt labels, and description."""
    parts: list[str] = []
    label = _clean_text(preferred_label)
    alt = _clean_text(alt_labels).replace("|", " ")
    desc = _clean_text(description)

    if label:
        parts.append(label)
    if alt:
        parts.append(alt)
    if desc:
        parts.append(desc)

    return " ".join(parts).strip()


def _read_csv(path: str, required_columns: list[str]) -> pd.DataFrame:
    """Read CSV and validate required columns."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"ESCO CSV not found: {path}")

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    return df


def build_index() -> dict[str, Any]:
    """Build ESCO embedding index and cache it with pickle.

    If cache file exists, loads and returns cached index.

    Returns:
        Dictionary containing occupation and skill embeddings with metadata.
    """
    cache_file = Path(CACHE_PATH)
    if cache_file.exists():
        with cache_file.open("rb") as f:
            return pickle.load(f)

    occupations_df = _read_csv(
        OCCUPATIONS_CSV,
        ["conceptUri", "preferredLabel", "altLabels", "description", "iscoGroup"],
    )
    skills_df = _read_csv(
        SKILLS_CSV,
        ["conceptUri", "preferredLabel", "altLabels", "description"],
    )

    occupation_texts = [
        _build_embedding_text(row.preferredLabel, row.altLabels, row.description)
        for row in occupations_df.itertuples(index=False)
    ]
    skill_texts = [
        _build_embedding_text(row.preferredLabel, row.altLabels, row.description)
        for row in skills_df.itertuples(index=False)
    ]

    model = _get_model()
    occupation_embeddings = model.encode(
        occupation_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    skill_embeddings = model.encode(
        skill_texts,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    index: dict[str, Any] = {
        "occupation_embeddings": occupation_embeddings,
        "occupation_uris": occupations_df["conceptUri"].astype(str).tolist(),
        "occupation_labels": occupations_df["preferredLabel"].astype(str).tolist(),
        "occupation_isco": occupations_df["iscoGroup"].astype(str).tolist(),
        "skill_embeddings": skill_embeddings,
        "skill_uris": skills_df["conceptUri"].astype(str).tolist(),
        "skill_labels": skills_df["preferredLabel"].astype(str).tolist(),
    }

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as f:
        pickle.dump(index, f)

    return index
