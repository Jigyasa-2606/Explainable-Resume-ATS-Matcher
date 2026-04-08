import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from src.extraction.router import extract_resume_text, looks_like_path
from src.preprocess.section_segmenter import segment_resume_sections
from src.preprocess.text_cleaning import clean_text


def _cache_key(resume_value: str) -> str:
    return hashlib.md5(resume_value.encode("utf-8")).hexdigest()


def _read_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    return json.loads(cache_path.read_text(encoding="utf-8"))


def _write_cache(cache_path: Path, cache: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, ensure_ascii=True, indent=2), encoding="utf-8")


def _resolve_resume_source(raw_resume_value: str, resumes_dir: Path) -> Dict[str, Any]:
    value = (raw_resume_value or "").strip()
    if not value:
        return {"raw_text": "", "source": "empty", "used_ocr": False}

    if looks_like_path(value):
        file_path = Path(value)
        if not file_path.is_absolute():
            file_path = resumes_dir / file_path
        extraction = extract_resume_text(str(file_path))
        return {"raw_text": extraction.text, "source": extraction.source, "used_ocr": extraction.used_ocr}

    return {"raw_text": value, "source": "inline_text", "used_ocr": False}


def build_phase1_dataset(
    input_csv: Path,
    output_csv: Path,
    cache_path: Path,
    resumes_dir: Path,
) -> None:
    df = pd.read_csv(input_csv)
    required_cols = {"job_description", "resume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    cache = _read_cache(cache_path)
    output_rows = []

    for _, row in df.iterrows():
        resume_value = str(row.get("resume", ""))
        job_desc = str(row.get("job_description", ""))
        row_key = _cache_key(resume_value)

        if row_key in cache:
            processed = cache[row_key]
        else:
            source_data = _resolve_resume_source(resume_value, resumes_dir=resumes_dir)
            raw_resume_text = source_data["raw_text"]
            cleaned_resume_text = clean_text(raw_resume_text)
            sections = segment_resume_sections(raw_resume_text)
            processed = {
                "resume_raw_text": raw_resume_text,
                "resume_clean_text": cleaned_resume_text,
                "section_skills": clean_text(sections.get("skills", "")),
                "section_experience": clean_text(sections.get("experience", "")),
                "section_education": clean_text(sections.get("education", "")),
                "section_projects": clean_text(sections.get("projects", "")),
                "extract_source": source_data["source"],
                "used_ocr": source_data["used_ocr"],
            }
            cache[row_key] = processed

        output_rows.append(
            {
                "job_description": job_desc,
                "job_description_clean": clean_text(job_desc),
                "resume": resume_value,
                "match_score": row.get("match_score"),
                **processed,
            }
        )

    out_df = pd.DataFrame(output_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    _write_cache(cache_path, cache)

    print(f"Wrote processed dataset: {output_csv}")
    print(f"Rows: {len(out_df)} | Cache entries: {len(cache)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 1 extraction + preprocessing dataset.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("resume_job_matching_dataset.csv"),
        help="CSV containing columns: job_description,resume[,match_score]",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("data/processed/phase1_processed.csv"),
        help="Output CSV path for processed rows.",
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("data/cache/extraction_cache.json"),
        help="JSON cache path to avoid reprocessing resumes.",
    )
    parser.add_argument(
        "--resumes-dir",
        type=Path,
        default=Path("."),
        help="Base directory for relative resume file paths in CSV.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_phase1_dataset(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        cache_path=args.cache_path,
        resumes_dir=args.resumes_dir,
    )
