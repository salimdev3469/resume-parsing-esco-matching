# CLI entrypoint for running CV-ESCO matching and printing/saving JSON output.

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pipeline import process_cv


def _build_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(description="CV-ESCO matcher")
    parser.add_argument("--cv", required=True, help="Path to CV file (.pdf or .docx)")
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Keep top K results for occupations and skills in final output",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output JSON path (e.g., result.json)",
    )
    return parser


def _trim_results(result: dict[str, Any], top_k: int) -> dict[str, Any]:
    """Trim occupation and skill lists to top_k and refresh stats."""
    top_k = max(1, int(top_k))
    result["occupations"] = result.get("occupations", [])[:top_k]
    result["skills"] = result.get("skills", [])[:top_k]

    stats = result.get("stats", {})
    stats["occupation_count"] = len(result["occupations"])
    stats["skill_count"] = len(result["skills"])
    result["stats"] = stats
    return result


def main() -> None:
    """Run CLI workflow."""
    parser = _build_parser()
    args = parser.parse_args()

    result = process_cv(args.cv)
    result = _trim_results(result, args.top_k)
    output_json = json.dumps(result, ensure_ascii=False, indent=2)

    print(output_json)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_json, encoding="utf-8")


if __name__ == "__main__":
    main()
