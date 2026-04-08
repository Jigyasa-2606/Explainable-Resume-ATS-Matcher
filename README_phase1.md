# Phase 1: Extraction + Preprocessing

## Install

```bash
pip install -r requirements.txt
```

System dependencies needed for OCR:
- Tesseract OCR binary (`tesseract`)
- Poppler tools (`pdftoppm`) for `pdf2image`

## Run

```bash
python -m src.pipeline.build_phase1_dataset \
  --input-csv resume_job_matching_dataset.csv \
  --output-csv data/processed/phase1_processed.csv \
  --cache-path data/cache/extraction_cache.json \
  --resumes-dir .
```

## CSV expectations

Required columns:
- `job_description`
- `resume`

`resume` can be:
- Inline resume text (current dataset style), or
- File path (`.pdf`, `.docx`, `.txt`) absolute or relative to `--resumes-dir`.

Output adds:
- `job_description_clean`
- `resume_raw_text`
- `resume_clean_text`
- `section_skills`
- `section_experience`
- `section_education`
- `section_projects`
- `extract_source`
- `used_ocr`

## Phase 2

```bash
python -m src.features.build_phase2_features \
  --input-csv data/processed/phase1_processed.csv \
  --output-dir data/features \
  --max-features 4000 \
  --ngram-max 2
```

## Phase 3

Train model:

```bash
python -m src.model.train_xgboost \
  --features-path data/features/X_features.npz \
  --labels-path data/features/y.npy \
  --output-dir artifacts/model
```

Predict one score:

```bash
python -m src.model.predict_score \
  --job-description "Data Scientist with Python SQL NLP and ML experience" \
  --resume-text "Worked with Python SQL Pandas NLP and machine learning projects for 3 years"
```

## Phase 4 (Explainability + Suggestions)

```bash
python -m src.explain.explain_predict \
  --model-path artifacts/model/xgb_regressor.joblib \
  --vectorizer-path data/features/tfidf_vectorizer.joblib \
  --feature-names-path data/features/feature_names.json \
  --job-description "Data Scientist with Python SQL NLP and ML experience" \
  --resume-text "Worked with Python SQL Pandas NLP and machine learning projects for 3 years" \
  --top-k 8
```

Output JSON includes:
- `match_score`
- `top_positive_features`
- `top_negative_features`
- `actionable_suggestions`

## Batch Ranking (One JD vs many resumes)

```bash
python -m src.pipeline.batch_rank_resumes \
  --input-csv data/processed/phase1_processed.csv \
  --output-csv artifacts/predictions/ranked_resumes.csv \
  --output-json artifacts/predictions/top_explanations.json \
  --job-description "Data Scientist with Python, SQL, NLP, and ML deployment experience" \
  --top-explain-count 20 \
  --shap-top-k 8
```

Outputs:
- Ranked CSV for all resumes by score
- JSON explanations + suggestions for top ranked candidates

## Streamlit UI

```bash
streamlit run app/streamlit_app.py
```

UI supports:
- Job description input
- Batch ranking run button
- Ranked candidate table
- Top-candidate feature explanations and suggestions
- Single-resume analysis and optional live job fetch links
