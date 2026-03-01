# FastAPI service layer for CV-ESCO matching using pipeline.process_cv.

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import process_cv

app = FastAPI(title="CV ESCO Matcher API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health-check endpoint."""
    return {"status": "ok"}


@app.post("/match")
async def match_cv(
    file: UploadFile = File(...),
    top_k: int = Form(default=10),
) -> dict[str, Any]:
    """Upload a CV file, run ESCO matching, and return results."""
    temp_path: str | None = None
    try:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in {".pdf", ".docx"}:
            raise HTTPException(
                status_code=400,
                detail="Unsupported CV format. Supported formats are .pdf and .docx.",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            temp_path = tmp_file.name

        return process_cv(temp_path, top_k=top_k)

    except HTTPException:
        raise
    except ValueError as exc:
        message = str(exc)
        if "Unsupported CV format" in message:
            raise HTTPException(status_code=400, detail=message) from exc
        raise HTTPException(status_code=500, detail=f"Matching failed: {message}") from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected server error: {exc}") from exc
    finally:
        try:
            await file.close()
        except Exception:
            pass
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port)
