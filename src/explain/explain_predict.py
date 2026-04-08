import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from src.explain.shap_explainer import explain_single_prediction
from src.explain.suggestions import generate_actionable_suggestions
from src.model.predict_score import build_inference_matrix


def run_explained_prediction(
    model_path: Path,
    vectorizer_path: Path,
    feature_names_path: Path,
    job_description: str,
    resume_text: str,
    top_k: int = 10,
):
    model = joblib.load(model_path)
    x_row, jd_clean, resume_clean = build_inference_matrix(
        vectorizer_path=vectorizer_path,
        job_description=job_description,
        resume_text=resume_text,
    )

    pred_score = float(np.clip(model.predict(x_row)[0], 0, 100))
    explanation = explain_single_prediction(
        model_path=model_path,
        feature_names_path=feature_names_path,
        x_row=x_row,
        top_k=top_k,
    )
    suggestions = generate_actionable_suggestions(
        job_description_clean=jd_clean,
        resume_clean_text=resume_clean,
        negative_features=explanation["negative"],
    )

    payload = {
        "match_score": round(pred_score, 2),
        "top_positive_features": explanation["positive"],
        "top_negative_features": explanation["negative"],
        "actionable_suggestions": suggestions,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict score with SHAP explanations and suggestions.")
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model/xgb_regressor.joblib"))
    parser.add_argument("--vectorizer-path", type=Path, default=Path("data/features/tfidf_vectorizer.joblib"))
    parser.add_argument("--feature-names-path", type=Path, default=Path("data/features/feature_names.json"))
    parser.add_argument("--job-description", type=str, required=True)
    parser.add_argument("--resume-text", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_explained_prediction(
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        feature_names_path=args.feature_names_path,
        job_description=args.job_description,
        resume_text=args.resume_text,
        top_k=args.top_k,
    )
    print(json.dumps(result, indent=2))
