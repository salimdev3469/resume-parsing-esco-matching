# End-to-end CV processing pipeline: read CV, extract chunks, match skills via embeddings and occupations via Gemini.

from __future__ import annotations

import json
import logging
import os
import re
from difflib import get_close_matches
from pathlib import Path
from typing import Any

import google.generativeai as genai
import pandas as pd

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
_ESCO_LOOKUP_CACHE: dict[str, list[str]] | None = None

_LOGGER = logging.getLogger(__name__)


def _get_gemini_api_key() -> str:
    """Return Gemini API key from environment or raise a clear error."""
    api_key = GEMINI_API_KEY.strip()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Define it as an environment variable before running."
        )
    return api_key


def _extract_json_array(raw_text: str) -> list[dict[str, Any]]:
    """Parse Gemini output into a JSON array of dictionaries."""
    text = (raw_text or "").strip()
    if not text:
        return []

    # Be tolerant to accidental wrappers while still expecting pure JSON array.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end >= start:
        text = text[start : end + 1]

    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Gemini response is not a JSON array.")
    return [item for item in data if isinstance(item, dict)]


def _estimate_chunk_count(text: str) -> int:
    """Estimate chunk-like phrase count without loading heavy NLP models."""
    if not text or not text.strip():
        return 0

    parts = re.split(r"[\n,;|]", text)
    seen: set[str] = set()
    for part in parts:
        normalized = " ".join(part.lower().split()).strip()
        if not normalized:
            continue
        wc = len(normalized.split())
        if 2 <= wc <= 6:
            seen.add(normalized)
    return len(seen)


def _load_esco_lookup() -> dict[str, list[str]]:
    """Load only ESCO labels/URIs needed for difflib mapping and cache in memory."""
    global _ESCO_LOOKUP_CACHE
    if _ESCO_LOOKUP_CACHE is not None:
        return _ESCO_LOOKUP_CACHE

    occupations_df = pd.read_csv(
        "data/occupations_en.csv",
        dtype=str,
        usecols=["conceptUri", "preferredLabel", "iscoGroup"],
    ).fillna("")
    skills_df = pd.read_csv(
        "data/skills_en.csv",
        dtype=str,
        usecols=["conceptUri", "preferredLabel"],
    ).fillna("")

    _ESCO_LOOKUP_CACHE = {
        "occupation_uris": occupations_df["conceptUri"].astype(str).tolist(),
        "occupation_labels": occupations_df["preferredLabel"].astype(str).tolist(),
        "occupation_isco": occupations_df["iscoGroup"].astype(str).tolist(),
        "skill_uris": skills_df["conceptUri"].astype(str).tolist(),
        "skill_labels": skills_df["preferredLabel"].astype(str).tolist(),
    }
    return _ESCO_LOOKUP_CACHE


def match_occupations_with_gemini(text: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Match occupations from full CV text using Gemini.

    Args:
        text: Raw CV text.
        top_k: Number of occupation candidates to request.

    Returns:
        Occupation candidates in standardized format.
    """
    if not text or not text.strip():
        return []

    genai.configure(api_key=_get_gemini_api_key())
    model = genai.GenerativeModel("gemini-1.5-flash")

    system_prompt = (
        "You are an expert in the ESCO v1.1 European Skills, Competences, Qualifications and "
        "Occupations taxonomy.\n"
        "Given a CV text, identify the most relevant ESCO occupations for this person.\n"
        "Return ONLY a valid JSON array, no explanation, no markdown, no code blocks.\n"
        "Each item must have exactly these fields:\n"
        '- "label": ESCO preferredLabel of the occupation (string)\n'
        '- "isco_group": 4-digit ISCO-08 group code (string)\n'
        '- "confidence": your confidence score between 0.0 and 1.0 (float)\n'
        '- "reasoning": one short sentence why this occupation fits (string)\n'
        f"Return the top {top_k} most relevant occupations, ordered by confidence descending."
    )

    try:
        response = model.generate_content(
            [
                {"role": "user", "parts": [system_prompt]},
                {"role": "user", "parts": [text]},
            ]
        )
        raw_text = response.text or ""
        parsed = _extract_json_array(raw_text)
    except Exception as exc:
        _LOGGER.exception("Gemini occupation matching failed: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    for item in parsed:
        label = str(item.get("label", "")).strip()
        if not label:
            continue

        confidence = item.get("confidence", 0.0)
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0

        isco_group = str(item.get("isco_group", "")).strip()
        reasoning = str(item.get("reasoning", "")).strip()

        results.append(
            {
                "uri": None,
                "label": label,
                "isco_group": isco_group,
                "score": score,
                "match_count": 1,
                "reasoning": reasoning,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[: max(1, int(top_k))]


def match_skills_with_gemini(text: str, top_k: int = 15) -> list[dict[str, Any]]:
    """Match skills from full CV text using Gemini.

    Args:
        text: Raw CV text.
        top_k: Number of skill candidates to request.

    Returns:
        Skill candidates in standardized format.
    """
    if not text or not text.strip():
        return []

    genai.configure(api_key=_get_gemini_api_key())
    model = genai.GenerativeModel("gemini-1.5-flash")

    system_prompt = (
        "You are an expert in the ESCO v1.1 European Skills, Competences, Qualifications and "
        "Occupations taxonomy.\n"
        "Given a CV text, identify the most relevant ESCO skills and competences this person has demonstrated.\n"
        "Focus on: technical skills, tools, programming languages, frameworks, methodologies, soft skills, domain knowledge.\n"
        "Return ONLY a valid JSON array, no explanation, no markdown, no code blocks.\n"
        "Each item must have exactly these fields:\n"
        '- "label": ESCO preferredLabel of the skill (string)\n'
        '- "confidence": your confidence score between 0.0 and 1.0 (float)\n'
        '- "reasoning": one short sentence why this skill is evident from the CV (string)\n'
        f"Return the top {top_k} most relevant skills, ordered by confidence descending."
    )

    try:
        response = model.generate_content(
            [
                {"role": "user", "parts": [system_prompt]},
                {"role": "user", "parts": [text]},
            ]
        )
        raw_text = response.text or ""
        parsed = _extract_json_array(raw_text)
    except Exception as exc:
        _LOGGER.exception("Gemini skill matching failed: %s", exc)
        return []

    results: list[dict[str, Any]] = []
    for item in parsed:
        label = str(item.get("label", "")).strip()
        if not label:
            continue

        confidence = item.get("confidence", 0.0)
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0

        reasoning = str(item.get("reasoning", "")).strip()
        results.append(
            {
                "uri": None,
                "label": label,
                "score": score,
                "match_count": 1,
                "reasoning": reasoning,
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[: max(1, int(top_k))]


def process_cv(file_path: str, top_k: int = 10) -> dict[str, Any]:
    """Process a CV file and return ESCO occupation/skill matches.

    Pipeline steps:
    1) extract_text
    2) estimate chunk count (lightweight)
    3) load ESCO lookup labels/URIs
    4) match_occupations_with_gemini
    5) match_skills_with_gemini
    6) map Gemini occupation labels to ESCO URI/ISCO via difflib
    7) map Gemini skill labels to ESCO URI via difflib

    Args:
        file_path: Path to CV file.
        top_k: Top K output size for Gemini occupation and skill calls.

    Returns:
        Result dictionary with matches and summary stats.
    """
    from cv_reader import extract_text

    text = extract_text(file_path)
    chunk_count = _estimate_chunk_count(text)
    index = _load_esco_lookup()

    occupations = match_occupations_with_gemini(text, top_k=top_k)
    skills = match_skills_with_gemini(text, top_k=top_k)

    for occupation in occupations:
        gemini_label = occupation.get("label", "")
        matches = get_close_matches(gemini_label, index["occupation_labels"], n=1, cutoff=0.6)
        if matches:
            idx = index["occupation_labels"].index(matches[0])
            occupation["uri"] = index["occupation_uris"][idx]
            occupation["isco_group"] = index["occupation_isco"][idx]

    for skill in skills:
        gemini_label = skill.get("label", "")
        matches = get_close_matches(gemini_label, index["skill_labels"], n=1, cutoff=0.5)
        if matches:
            idx = index["skill_labels"].index(matches[0])
            skill["uri"] = index["skill_uris"][idx]

    return {
        "file": Path(file_path).name,
        "chunk_count": chunk_count,
        "occupations": occupations,
        "skills": skills,
        "stats": {
            "occupation_count": len(occupations),
            "skill_count": len(skills),
        },
    }
