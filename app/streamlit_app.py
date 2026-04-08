import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure project root is importable when launched via streamlit.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.batch_rank_resumes import batch_rank
from src.explain.explain_predict import run_explained_prediction
from src.extraction.router import extract_resume_text
from src.jobs.live_jobs import fetch_live_jobs
from src.model.predict_score import predict_match_score
from src.skills.skill_gap import analyze_skill_gap, top_missing_skill_suggestions


st.set_page_config(page_title="Resume Matcher", layout="wide")
st.title("Resume Matcher")

default_input_csv = "data/processed/phase1_processed.csv"
default_output_csv = "artifacts/predictions/ranked_resumes.csv"
default_output_json = "artifacts/predictions/top_explanations.json"
default_model_path = "artifacts/model/xgb_regressor.joblib"
default_vectorizer_path = "data/features/tfidf_vectorizer.joblib"
default_feature_names_path = "data/features/feature_names.json"

jd_templates = {
    "None": "",
    "Data Scientist": "Looking for a Data Scientist with Python, SQL, machine learning, NLP, deep learning, and experiment tracking experience.",
    "ML Engineer": "Need an ML Engineer with Python, MLOps, TensorFlow or PyTorch, cloud deployment, and model monitoring skills.",
    "Data Analyst": "Need a Data Analyst with SQL, Excel, Tableau or Power BI, reporting, and data cleaning experience.",
    "Software Engineer": "Looking for a Software Engineer with Java or Python, system design, REST APIs, Docker, Git, and Agile experience.",
}

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose page",
        [
            "1) ATS Score Checker",
            "2) Resume to Job Postings",
            "3) Custom JD Explainability",
            "4) Matched vs Missing Skills",
        ],
    )

    st.header("Model Paths")
    st.header("Paths")
    input_csv = st.text_input("Input CSV", value=default_input_csv)
    output_csv = st.text_input("Output ranked CSV", value=default_output_csv)
    output_json = st.text_input("Output explanations JSON", value=default_output_json)
    model_path = st.text_input("Model path", value=default_model_path)
    vectorizer_path = st.text_input("Vectorizer path", value=default_vectorizer_path)
    feature_names_path = st.text_input("Feature names path", value=default_feature_names_path)
    top_explain_count = st.slider("Top candidates to explain", min_value=5, max_value=100, value=20, step=5)
    shap_top_k = st.slider("Top SHAP features per candidate", min_value=3, max_value=15, value=8, step=1)

if "job_description" not in st.session_state:
    st.session_state["job_description"] = ""

use_custom_jd_only = st.toggle("Use custom JD only", value=True)
template_name = st.selectbox("JD Template", list(jd_templates.keys()), index=0)
if not use_custom_jd_only:
    if st.button("Apply Selected Template"):
        st.session_state["job_description"] = jd_templates.get(template_name, "")

job_description = st.text_area(
    "Job Description",
    key="job_description",
    height=200,
    placeholder="Paste job description here...",
)

uploaded_resume = st.file_uploader(
    "Upload resume (PDF, DOCX, TXT)",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=False,
)

def _extract_uploaded_resume(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower() or ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        temp_path = Path(tmp.name)
    try:
        return extract_resume_text(str(temp_path))
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


if page == "1) ATS Score Checker":
    st.subheader("ATS Score Checker")
    run_ats = st.button("Check ATS Score", type="primary")
    if run_ats:
        if uploaded_resume is None:
            st.error("Please upload a resume.")
        elif not job_description.strip():
            st.error("Please provide a job description.")
        else:
            try:
                extracted = _extract_uploaded_resume(uploaded_resume)
                score = predict_match_score(
                    model_path=Path(model_path),
                    vectorizer_path=Path(vectorizer_path),
                    job_description=job_description,
                    resume_text=extracted.text,
                )
                st.metric("ATS Match Score", f"{score:.2f}/100")
                st.caption(f"Extraction source: {extracted.source} | OCR used: {extracted.used_ocr}")
            except Exception as exc:
                st.exception(exc)

elif page == "2) Resume to Job Postings":
    st.subheader("Resume to Job Postings")
    run_jobs = st.button("Find Jobs for Resume", type="primary")
    if run_jobs:
        if uploaded_resume is None:
            st.error("Please upload a resume.")
        else:
            try:
                extracted = _extract_uploaded_resume(uploaded_resume)
                live = fetch_live_jobs(
                    resume_text=extracted.text,
                    job_description=job_description,
                    per_source_limit=10,
                )
                st.caption(f"Search query used: {live.get('query', '')}")
                if live.get("errors"):
                    for err in live["errors"]:
                        st.warning(err)
                jobs = live.get("jobs", [])
                if not jobs:
                    st.info("No live jobs found. Try broader JD keywords.")
                for job in jobs[:25]:
                    st.markdown(
                        f"- **{job['title']}** at **{job['company']}** ({job['location']}) - "
                        f"[Apply Link]({job['url']}) | Source: {job['source']}"
                    )
            except Exception as exc:
                st.exception(exc)

elif page == "3) Custom JD Explainability":
    st.subheader("Custom JD + Resume Explainability")
    run_explain = st.button("Analyze with Explainability", type="primary")
    if run_explain:
        if uploaded_resume is None:
            st.error("Please upload a resume.")
        elif not job_description.strip():
            st.error("Please provide a job description.")
        else:
            try:
                extracted = _extract_uploaded_resume(uploaded_resume)
                result = run_explained_prediction(
                    model_path=Path(model_path),
                    vectorizer_path=Path(vectorizer_path),
                    feature_names_path=Path(feature_names_path),
                    job_description=job_description,
                    resume_text=extracted.text,
                    top_k=shap_top_k,
                )
                st.metric("Match Score", f"{result['match_score']}/100")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Top Positive Factors**")
                    st.json(result.get("top_positive_features", []))
                with col2:
                    st.markdown("**Top Negative Factors**")
                    st.json(result.get("top_negative_features", []))
                st.markdown("**Actionable Suggestions**")
                for suggestion in result.get("actionable_suggestions", []):
                    st.write(f"- {suggestion}")
            except Exception as exc:
                st.exception(exc)

elif page == "4) Matched vs Missing Skills":
    st.subheader("Matched and Missing Skills")
    run_gap = st.button("Analyze Skill Gap", type="primary")
    if run_gap:
        if uploaded_resume is None:
            st.error("Please upload a resume.")
        elif not job_description.strip():
            st.error("Please provide a job description.")
        else:
            try:
                extracted = _extract_uploaded_resume(uploaded_resume)
                gap = analyze_skill_gap(job_description=job_description, resume_text=extracted.text)
                st.metric("Skill Coverage", f"{gap['coverage_percent']}%")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Matched Skills**")
                    st.write(", ".join(gap["matched_skills"]) if gap["matched_skills"] else "None")
                with c2:
                    st.markdown("**Missing Skills**")
                    st.write(", ".join(gap["missing_skills"]) if gap["missing_skills"] else "None")
                st.markdown("**Recommended Improvements**")
                for suggestion in top_missing_skill_suggestions(gap["missing_skills"]):
                    st.write(f"- {suggestion}")
            except Exception as exc:
                st.exception(exc)

# Keep existing batch workflow as optional section at bottom.
with st.expander("Optional: Batch Ranking Existing Dataset"):
    run_batch = st.button("Run Batch Ranking")
    if run_batch:
        if not job_description.strip():
            st.error("Please provide a job description.")
        else:
            try:
                with st.spinner("Ranking resumes and generating explanations..."):
                    batch_rank(
                        input_csv=Path(input_csv),
                        output_csv=Path(output_csv),
                        output_json=Path(output_json),
                        model_path=Path(model_path),
                        vectorizer_path=Path(vectorizer_path),
                        feature_names_path=Path(feature_names_path),
                        job_description=job_description,
                        top_explain_count=top_explain_count,
                        shap_top_k=shap_top_k,
                    )
                st.success("Batch ranking completed.")
            except Exception as exc:
                st.exception(exc)

    if Path(output_csv).exists():
        ranked_df = pd.read_csv(output_csv)
        st.dataframe(ranked_df.head(50), use_container_width=True)
