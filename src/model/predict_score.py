import argparse
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.features.build_phase2_features import SKILL_PHRASES, _extract_years_signal, _skills_in_text
from src.preprocess.text_cleaning import clean_text


def _manual_feature_row(job_desc_clean: str, resume_clean: str) -> np.ndarray:
    jd_skills = _skills_in_text(job_desc_clean, SKILL_PHRASES)
    resume_skills = _skills_in_text(resume_clean, SKILL_PHRASES)
    overlap = jd_skills.intersection(resume_skills)

    skill_overlap_ratio = len(overlap) / max(len(jd_skills.union(resume_skills)), 1)
    jd_coverage_ratio = len(overlap) / max(len(jd_skills), 1)

    jd_tokens = set(job_desc_clean.split())
    resume_tokens = set(resume_clean.split())
    keyword_overlap_ratio = len(jd_tokens.intersection(resume_tokens)) / max(len(jd_tokens), 1)

    jd_years = _extract_years_signal(job_desc_clean)
    resume_years = _extract_years_signal(resume_clean)
    experience_year_gap_abs = abs(jd_years - resume_years)

    if job_desc_clean and resume_clean:
        local_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        local_matrix = local_vec.fit_transform([job_desc_clean, resume_clean])
        jd_resume_cos = float(cosine_similarity(local_matrix[0], local_matrix[1])[0][0])
    else:
        jd_resume_cos = 0.0

    return np.array(
        [[skill_overlap_ratio, jd_coverage_ratio, keyword_overlap_ratio, experience_year_gap_abs, jd_resume_cos]],
        dtype=np.float32,
    )


def predict_match_score(model_path: Path, vectorizer_path: Path, job_description: str, resume_text: str) -> float:
    model = joblib.load(model_path)
    tfidf = joblib.load(vectorizer_path)

    jd_clean = clean_text(job_description)
    resume_clean = clean_text(resume_text)

    combined = [f"{jd_clean} {resume_clean}"]
    x_tfidf = tfidf.transform(combined)
    x_manual = csr_matrix(_manual_feature_row(jd_clean, resume_clean))
    x = hstack([x_tfidf, x_manual], format="csr")

    score = float(model.predict(x)[0])
    return float(np.clip(score, 0, 100))


def build_inference_matrix(vectorizer_path: Path, job_description: str, resume_text: str):
    tfidf = joblib.load(vectorizer_path)
    jd_clean = clean_text(job_description)
    resume_clean = clean_text(resume_text)
    combined = [f"{jd_clean} {resume_clean}"]
    x_tfidf = tfidf.transform(combined)
    x_manual = csr_matrix(_manual_feature_row(jd_clean, resume_clean))
    x = hstack([x_tfidf, x_manual], format="csr")
    return x, jd_clean, resume_clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict resume-job match score.")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model/xgb_regressor.joblib"))
    parser.add_argument("--vectorizer-path", type=Path, default=Path("data/features/tfidf_vectorizer.joblib"))
    parser.add_argument("--job-description", type=str, required=True)
    parser.add_argument("--resume-text", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pred = predict_match_score(
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        job_description=args.job_description,
        resume_text=args.resume_text,
    )
    print(f"Predicted match score: {pred:.2f}")
