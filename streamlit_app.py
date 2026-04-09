import json
import sys
import tempfile
from pathlib import Path
import pandas as pd
import streamlit as st

# Fix missing PROJECT_ROOT (assuming current dir)
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.batch_rank_resumes import batch_rank
from src.explain.explain_predict import run_explained_prediction
from src.extraction.router import extract_resume_text

st.set_page_config(page_title="Resume Matcher", layout="wide")

st.title("Resume Matcher - Batch Screening")

default_input_csv = "data/processed/phase1_processed.csv"
default_output_csv = "artifacts/predictions/ranked_resumes.csv"
default_output_json = "artifacts/predictions/explanations.json"
default_vectorizer_path = "data/features/tfidf_vectorizer.joblib"
default_feature_names_path = "data/features/feature_names.json"
default_model_path = "artifacts/model/xgb_regressor.joblib"

jd_templates = {
    "None": "",
    "Data Scientist": "Looking for a Data Scientist with Python, SQL, machine learning, NLP, deep learning, and experiment tracking experience.",
    "ML Engineer": "Need an ML Engineer with Python, MLOps, TensorFlow or PyTorch, cloud deployment, and model monitoring skills.",
    "Data Analyst": "Need a Data Analyst with SQL, Excel, Tableau or Power BI, reporting, and data cleaning experience.",
    "Software Engineer": "Looking for a Software Engineer with Java or Python, system design, REST APIs, Docker, Git, and Agile experience.",
}

with st.sidebar:
    st.header("Paths")
    input_csv = st.text_input("Input CSV", value=default_input_csv)
    output_csv = st.text_input("Output CSV", value=default_output_csv)
    output_json = st.text_input("Output JSON", value=default_output_json)
    model_path = st.text_input("Model Path", value=default_model_path)
    vectorizer_path = st.text_input("Vectorizer Path", value=default_vectorizer_path)
    feature_names_path = st.text_input("Feature Names Path", value=default_feature_names_path)

    top_explain_count = st.slider("Top candidates to explain", 5, 100, 20, 5)
    shap_top_k = st.slider("Top SHAP features per candidate", 3, 15, 8, 1)

template_name = st.selectbox("JD Template", list(jd_templates.keys()))
job_description = st.text_area(
    "Job Description",
    value=jd_templates[template_name],
    height=200,
    placeholder="Paste job description here..."
)

run_clicked = st.button("Run Batch Ranking", type="primary")

single_tab, batch_tab = st.tabs(["Single Resume Upload", "Batch Ranking"])

# =========================
# BATCH RUN
# =========================
if run_clicked:
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

# =========================
# SINGLE RESUME TAB
# =========================
with single_tab:
    st.subheader("Analyze One Uploaded Resume")

    uploaded_resume = st.file_uploader(
        "Upload resume (PDF, DOCX, TXT)",
        type=["pdf", "docx", "txt"]
    )

    run_single = st.button("Analyze Uploaded Resume", type="primary")

    if run_single:
        if not job_description.strip():
            st.error("Please provide a job description.")
        elif uploaded_resume is None:
            st.error("Please upload a resume file.")
        else:
            try:
                suffix = Path(uploaded_resume.name).suffix.lower() or ".txt"

                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_resume.getbuffer())
                    temp_path = Path(tmp.name)

                with st.spinner("Extracting resume text and generating explanation..."):
                    extracted = extract_resume_text(str(temp_path))

                    result = run_explained_prediction(
                        model_path=Path(model_path),
                        vectorizer_path=Path(vectorizer_path),
                        feature_names_path=Path(feature_names_path),
                        job_description=job_description,
                        resume_text=extracted.text,
                        top_k=shap_top_k,
                    )

                st.success("Analysis completed.")
                st.metric("Match Score", f"{result['match_score']}/100")
                st.caption(f"Extraction source: {extracted.source} | OCR used: {extracted.used_ocr}")

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
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except:
                    pass

# =========================
# BATCH TAB VIEW
# =========================
with batch_tab:
    st.subheader("Rank Existing Dataset Resumes")

    if Path(output_csv).exists():
        st.markdown("**Ranked Candidates**")
        ranked_df = pd.read_csv(output_csv)
        st.dataframe(ranked_df.head(100), use_container_width=True)
    else:
        st.info("No ranked CSV found yet.")

    if Path(output_json).exists():
        st.markdown("**Top Candidate Explanations**")

        payload = json.loads(Path(output_json).read_text())
        explanations = payload.get("explanations", [])

        if not explanations:
            st.info("No explanations found.")

        for idx, item in enumerate(explanations[:10], start=1):
            with st.expander(f"Candidate {idx} | row_id={item.get('row_id')}"):
                st.code(item.get("resume_preview", ""))

                st.markdown("**Top Positive Features**")
                st.json(item.get("top_positive_features", []))

                st.markdown("**Top Negative Features**")
                st.json(item.get("top_negative_features", []))

                st.markdown("**Actionable Suggestions**")
                for suggestion in item.get("actionable_suggestions", []):
                    st.write(f"- {suggestion}")
    else:
        st.info("No explanation JSON found yet.")