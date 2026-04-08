import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz
import pytesseract
from pdf2image import convert_from_path

try:
    import docx2txt
except ImportError:  # pragma: no cover
    docx2txt = None


@dataclass
class ExtractionResult:
    text: str
    source: str
    used_ocr: bool


def _is_text_noisy(text: str, min_chars: int = 200, min_alpha_ratio: float = 0.55) -> bool:
    if not text:
        return True
    cleaned = text.strip()
    if len(cleaned) < min_chars:
        return True
    alpha_count = sum(ch.isalpha() for ch in cleaned)
    ratio = alpha_count / max(len(cleaned), 1)
    return ratio < min_alpha_ratio


def _extract_pdf_text_pymupdf(file_path: Path) -> str:
    text_parts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts)


def _extract_pdf_text_ocr(file_path: Path, dpi: int = 300, max_pages: Optional[int] = None) -> str:
    pages = convert_from_path(str(file_path), dpi=dpi)
    if max_pages is not None:
        pages = pages[:max_pages]
    text_parts = [pytesseract.image_to_string(page) for page in pages]
    return "\n".join(text_parts)


def _extract_docx_text(file_path: Path) -> str:
    if docx2txt is None:
        raise ImportError("docx2txt is required for DOCX extraction. Run: pip install docx2txt")
    return docx2txt.process(str(file_path)) or ""


def extract_resume_text(file_path: str) -> ExtractionResult:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Resume file does not exist: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        parsed_text = _extract_pdf_text_pymupdf(path)
        if _is_text_noisy(parsed_text):
            ocr_text = _extract_pdf_text_ocr(path)
            return ExtractionResult(text=ocr_text, source="pdf_ocr", used_ocr=True)
        return ExtractionResult(text=parsed_text, source="pdf_text", used_ocr=False)

    if suffix == ".docx":
        text = _extract_docx_text(path)
        return ExtractionResult(text=text, source="docx_text", used_ocr=False)

    if suffix == ".txt":
        return ExtractionResult(text=path.read_text(encoding="utf-8", errors="ignore"), source="txt", used_ocr=False)

    raise ValueError(f"Unsupported file type: {suffix}")


def looks_like_path(value: str) -> bool:
    if not isinstance(value, str):
        return False
    trimmed = value.strip()
    if not trimmed:
        return False
    return bool(re.search(r"\.(pdf|docx|txt)$", trimmed, flags=re.IGNORECASE))
