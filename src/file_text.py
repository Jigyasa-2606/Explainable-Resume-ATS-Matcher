from __future__ import annotations

from pathlib import Path


def extract_text_from_upload(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".txt":
        return content.decode("utf-8", errors="ignore")

    if suffix == ".pdf":
        try:
            from io import BytesIO

            from pypdf import PdfReader  # type: ignore[import-not-found]

            reader = PdfReader(BytesIO(content))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""

    if suffix == ".docx":
        try:
            from io import BytesIO

            from docx import Document  # type: ignore[import-not-found]

            document = Document(BytesIO(content))
            return "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception:
            return ""

    return ""
