import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from src.explain.shap_explainer import explain_single_prediction_loaded, load_feature_names
from src.explain.suggestions import generate_actionable_suggestions
from src.model.predict_score import build_inference_matrix


def _safe_text(value) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower() == "nan":
        return ""
    return text


def batch_rank(
    input_csv: Path,
    output_csv: Path,
    output_json: Path,
    model_path: Path,
    vectorizer_path: Path,
    feature_names_path: Path,
    job_description: str,
    top_explain_count: int = 20,
    shap_top_k: int = 8,
) -> None:
    df = pd.read_csv(input_csv)
    if "resume_raw_text" not in df.columns:
        raise ValueError("Input CSV must include 'resume_raw_text' (use Phase 1 processed CSV).")

    model = joblib.load(model_path)
    feature_names = load_feature_names(feature_names_path)

    rows: List[Dict[str, object]] = []
    explain_payload: List[Dict[str, object]] = []

    for idx, row in df.iterrows():
        resume_text = _safe_text(row.get("resume_raw_text", ""))
        x_row, jd_clean, resume_clean = build_inference_matrix(
            vectorizer_path=vectorizer_path,
            job_description=job_description,
            resume_text=resume_text,
        )
        score = float(np.clip(model.predict(x_row)[0], 0, 100))

        rows.append(
            {
                "row_id": idx,
                "score": round(score, 4),
                "resume_preview": resume_text[:200],
                "extract_source": _safe_text(row.get("extract_source", "")),
                "used_ocr": _safe_text(row.get("used_ocr", "")),
            }
        )

        if len(explain_payload) < top_explain_count:
            explain_payload.append(
                {
                    "row_id": idx,
                    "x_row": x_row,
                    "jd_clean": jd_clean,
                    "resume_clean": resume_clean,
                    "resume_preview": resume_text[:200],
                }
            )

    ranked = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(output_csv, index=False)

    top_ids = set(ranked.head(top_explain_count)["row_id"].tolist())
    explanations: List[Dict[str, object]] = []
    for item in explain_payload:
        if item["row_id"] not in top_ids:
            continue
        exp = explain_single_prediction_loaded(
            model=model,
            feature_names=feature_names,
            x_row=item["x_row"],
            top_k=shap_top_k,
        )
        suggestions = generate_actionable_suggestions(
            job_description_clean=item["jd_clean"],
            resume_clean_text=item["resume_clean"],
            negative_features=exp["negative"],
        )
        explanations.append(
            {
                "row_id": item["row_id"],
                "resume_preview": item["resume_preview"],
                "top_positive_features": exp["positive"],
                "top_negative_features": exp["negative"],
                "actionable_suggestions": suggestions,
            }
        )

    payload = {
        "job_description": job_description,
        "ranked_csv": str(output_csv),
        "top_explain_count": top_explain_count,
        "explanations": explanations,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved ranked results: {output_csv}")
    print(f"Saved top explanations: {output_json}")
    print(f"Total resumes ranked: {len(ranked)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch rank resumes for a single JD.")
    parser.add_argument("--input-csv", type=Path, default=Path("data/processed/phase1_processed.csv"))
    parser.add_argument("--output-csv", type=Path, default=Path("artifacts/predictions/ranked_resumes.csv"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/predictions/top_explanations.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model/xgb_regressor.joblib"))
    parser.add_argument("--vectorizer-path", type=Path, default=Path("data/features/tfidf_vectorizer.joblib"))
    parser.add_argument("--feature-names-path", type=Path, default=Path("data/features/feature_names.json"))
    parser.add_argument("--job-description", type=str, required=True)
    parser.add_argument("--top-explain-count", type=int, default=20)
    parser.add_argument("--shap-top-k", type=int, default=8)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    batch_rank(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        output_json=args.output_json,
        model_path=args.model_path,
        vectorizer_path=args.vectorizer_path,
        feature_names_path=args.feature_names_path,
        job_description=args.job_description,
        top_explain_count=args.top_explain_count,
        shap_top_k=args.shap_top_k,
    )
