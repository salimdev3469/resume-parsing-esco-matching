# Reads CV files (PDF, DOCX) and returns plain text content.

from __future__ import annotations

from pathlib import Path

import pdfplumber
from docx import Document


def extract_text(file_path: str) -> str:
    """Extract plain text from a CV file.

    Supports PDF and DOCX files.

    Args:
        file_path: Path to the CV file.

    Returns:
        Extracted text as a single string.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is not supported.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CV file not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        pages: list[str] = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text.strip())
        return "\n".join(pages).strip()

    if suffix == ".docx":
        doc = Document(path)
        lines = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
        return "\n".join(lines).strip()

    raise ValueError(
        f"Unsupported CV format: {suffix or 'unknown'}. Supported formats are .pdf and .docx."
    )
