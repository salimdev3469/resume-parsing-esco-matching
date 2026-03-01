# Matches extracted CV chunks against ESCO occupation and skill embeddings.

from __future__ import annotations

from collections import defaultdict
from typing import Any, Literal

import torch
from sentence_transformers import SentenceTransformer, util

MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer model globally."""
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def _as_tensor(value: Any) -> torch.Tensor:
    """Convert cached embedding matrix to torch.Tensor if needed."""
    if isinstance(value, torch.Tensor):
        return value
    return torch.tensor(value)


def _aggregate_matches(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate chunk-level matches by URI using average score and match count."""
    grouped: dict[str, dict[str, Any]] = {}
    scores_by_uri: dict[str, list[float]] = defaultdict(list)

    for item in candidates:
        uri = item["uri"]
        scores_by_uri[uri].append(float(item["score"]))
        if uri not in grouped:
            grouped[uri] = {k: v for k, v in item.items() if k != "score"}

    aggregated: list[dict[str, Any]] = []
    for uri, base in grouped.items():
        scores = scores_by_uri[uri]
        result = dict(base)
        result["score"] = sum(scores) / len(scores)
        result["match_count"] = len(scores)
        aggregated.append(result)

    aggregated.sort(key=lambda x: (x["score"], x["match_count"]), reverse=True)
    return aggregated


def match_chunks(
    chunks: list[str],
    index: dict[str, Any],
    top_k_per_chunk: int = 3,
    occupation_threshold: float = 0.35,
    skill_threshold: float = 0.35,
    mode: Literal["both", "skills_only"] = "both",
) -> dict[str, list[dict[str, Any]]]:
    """Match CV chunks against ESCO occupations and skills.

    Args:
        chunks: Extracted chunk list.
        index: ESCO index dictionary from build_index().
        top_k_per_chunk: Maximum matches per chunk per category.
        occupation_threshold: Minimum cosine score for occupation candidates.
        skill_threshold: Minimum cosine score for skill candidates.
        mode: "both" to match occupations and skills, "skills_only" to skip occupations.

    Returns:
        Dictionary with aggregated "occupations" and "skills" matches.
    """
    if not chunks:
        return {"occupations": [], "skills": []}

    model = _get_model()

    chunk_embeddings = model.encode(
        chunks,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    skill_embeddings = _as_tensor(index["skill_embeddings"])

    occupation_embeddings: torch.Tensor | None = None
    occupation_uris: list[str] = []
    occupation_labels: list[str] = []
    occupation_isco: list[str] = []
    if mode != "skills_only":
        occupation_embeddings = _as_tensor(index["occupation_embeddings"])
        occupation_uris = index["occupation_uris"]
        occupation_labels = index["occupation_labels"]
        occupation_isco = index["occupation_isco"]

    skill_uris: list[str] = index["skill_uris"]
    skill_labels: list[str] = index["skill_labels"]

    occupation_candidates: list[dict[str, Any]] = []
    skill_candidates: list[dict[str, Any]] = []

    for chunk_emb in chunk_embeddings:
        if mode != "skills_only" and occupation_embeddings is not None:
            occ_sims = util.cos_sim(chunk_emb, occupation_embeddings)[0]
            occ_top_k = min(top_k_per_chunk, occ_sims.shape[0])
            occ_vals, occ_idx = torch.topk(occ_sims, k=occ_top_k)
            for score_tensor, idx_tensor in zip(occ_vals, occ_idx):
                score = float(score_tensor.item())
                if score < occupation_threshold:
                    continue
                idx = int(idx_tensor.item())
                occupation_candidates.append(
                    {
                        "uri": occupation_uris[idx],
                        "label": occupation_labels[idx],
                        "isco_group": occupation_isco[idx],
                        "score": score,
                    }
                )

        skill_sims = util.cos_sim(chunk_emb, skill_embeddings)[0]
        skill_top_k = min(top_k_per_chunk, skill_sims.shape[0])
        skill_vals, skill_idx = torch.topk(skill_sims, k=skill_top_k)
        for score_tensor, idx_tensor in zip(skill_vals, skill_idx):
            score = float(score_tensor.item())
            if score < skill_threshold:
                continue
            idx = int(idx_tensor.item())
            skill_candidates.append(
                {
                    "uri": skill_uris[idx],
                    "label": skill_labels[idx],
                    "score": score,
                }
            )

    occupations = _aggregate_matches(occupation_candidates) if mode != "skills_only" else []
    skills = _aggregate_matches(skill_candidates)

    return {"occupations": occupations, "skills": skills}
