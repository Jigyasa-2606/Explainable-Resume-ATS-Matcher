import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SKILL_PHRASES = [
    "python",
    "sql",
    "excel",
    "tableau",
    "power bi",
    "machine learning",
    "deep learning",
    "nlp",
    "pandas",
    "tensorflow",
    "pytorch",
    "keras",
    "mlops",
    "computer vision",
    "cloud",
    "aws",
    "gcp",
    "azure",
    "java",
    "spring boot",
    "docker",
    "git",
    "agile",
    "rest apis",
    "system design",
    "stakeholder management",
    "product roadmap",
    "user stories",
    "scrum",
    "reporting",
    "data cleaning",
    "etl",
]


def _safe_text(value: object) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def _extract_years_signal(text: str) -> float:
    # Signals patterns like "3 years", "5+ years", "2 yrs".
    matches = re.findall(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)", text.lower())
    if not matches:
        return 0.0
    years = [float(m) for m in matches]
    return max(years)


def _skills_in_text(text: str, skill_phrases: Iterable[str]) -> Set[str]:
    lowered = text.lower()
    found = set()
    for skill in skill_phrases:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, lowered):
            found.add(skill)
    return found


def _build_manual_features(df: pd.DataFrame, skill_phrases: Iterable[str]) -> Tuple[np.ndarray, List[str]]:
    feature_rows = []
    feature_names = [
        "skill_overlap_ratio",
        "skill_jd_coverage_ratio",
        "keyword_overlap_ratio",
        "experience_year_gap_abs",
        "jd_resume_cosine_clean",
    ]

    for _, row in df.iterrows():
        jd_clean = _safe_text(row.get("job_description_clean", ""))
        resume_clean = _safe_text(row.get("resume_clean_text", ""))
        resume_sections = " ".join(
            [
                _safe_text(row.get("section_skills", "")),
                _safe_text(row.get("section_experience", "")),
                _safe_text(row.get("section_education", "")),
                _safe_text(row.get("section_projects", "")),
            ]
        ).strip()
        resume_view = resume_sections if resume_sections else resume_clean

        jd_skills = _skills_in_text(jd_clean, skill_phrases)
        resume_skills = _skills_in_text(resume_view, skill_phrases)
        overlap = jd_skills.intersection(resume_skills)
        skill_overlap_ratio = len(overlap) / max(len(jd_skills.union(resume_skills)), 1)
        jd_coverage_ratio = len(overlap) / max(len(jd_skills), 1)

        jd_tokens = set(jd_clean.split())
        resume_tokens = set(resume_view.split())
        keyword_overlap_ratio = len(jd_tokens.intersection(resume_tokens)) / max(len(jd_tokens), 1)

        jd_years = _extract_years_signal(jd_clean)
        resume_years = _extract_years_signal(resume_view)
        experience_year_gap_abs = abs(jd_years - resume_years)

        if jd_clean and resume_view:
            local_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
            local_matrix = local_vec.fit_transform([jd_clean, resume_view])
            jd_resume_cos = float(cosine_similarity(local_matrix[0], local_matrix[1])[0][0])
        else:
            jd_resume_cos = 0.0

        feature_rows.append(
            [
                skill_overlap_ratio,
                jd_coverage_ratio,
                keyword_overlap_ratio,
                experience_year_gap_abs,
                jd_resume_cos,
            ]
        )

    return np.array(feature_rows, dtype=np.float32), feature_names


def build_phase2_features(
    input_csv: Path,
    output_dir: Path,
    max_features: int = 4000,
    ngram_max: int = 2,
) -> None:
    df = pd.read_csv(input_csv)
    required_cols = {"job_description_clean", "resume_clean_text"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for Phase 2: {missing}")

    df["job_description_clean"] = df["job_description_clean"].fillna("")
    df["resume_clean_text"] = df["resume_clean_text"].fillna("")

    combined_text = (df["job_description_clean"] + " " + df["resume_clean_text"]).tolist()
    tfidf = TfidfVectorizer(ngram_range=(1, ngram_max), max_features=max_features, min_df=2)
    x_tfidf = tfidf.fit_transform(combined_text)

    manual_np, manual_feature_names = _build_manual_features(df, SKILL_PHRASES)
    x_manual = csr_matrix(manual_np)

    x_all = hstack([x_tfidf, x_manual], format="csr")
    y = df["match_score"].values if "match_score" in df.columns else None

    output_dir.mkdir(parents=True, exist_ok=True)
    save_npz(output_dir / "X_features.npz", x_all)
    np.save(output_dir / "y.npy", y)
    joblib.dump(tfidf, output_dir / "tfidf_vectorizer.joblib")

    tfidf_feature_names = tfidf.get_feature_names_out().tolist()
    all_feature_names = tfidf_feature_names + manual_feature_names
    (output_dir / "feature_names.json").write_text(json.dumps(all_feature_names, indent=2), encoding="utf-8")
    (output_dir / "manual_feature_names.json").write_text(json.dumps(manual_feature_names, indent=2), encoding="utf-8")

    meta: Dict[str, object] = {
        "rows": int(x_all.shape[0]),
        "tfidf_features": int(x_tfidf.shape[1]),
        "manual_features": int(x_manual.shape[1]),
        "total_features": int(x_all.shape[1]),
        "max_tfidf_features": int(max_features),
        "ngram_max": int(ngram_max),
    }
    (output_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Saved feature matrix to: {output_dir / 'X_features.npz'}")
    print(f"Saved labels to: {output_dir / 'y.npy'}")
    print(f"Rows={meta['rows']} TFIDF={meta['tfidf_features']} Manual={meta['manual_features']} Total={meta['total_features']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 2 TF-IDF + manual features.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("data/processed/phase1_processed.csv"),
        help="Processed CSV from Phase 1.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/features"),
        help="Output directory for feature artifacts.",
    )
    parser.add_argument("--max-features", type=int, default=4000, help="Max TF-IDF features.")
    parser.add_argument("--ngram-max", type=int, default=2, help="Upper bound for ngram range.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_phase2_features(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        max_features=args.max_features,
        ngram_max=args.ngram_max,
    )
