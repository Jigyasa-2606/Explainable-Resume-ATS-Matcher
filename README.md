# AI-Powered Resume Analyzer

This project analyzes a resume against a job description using semantic similarity,
context-aware skill matching, section quality checks, ATS scoring, and bias/fairness
analysis.

## Setup

```bash
pip install -r requirements.txt
```

## Deployment

For Vercel + Render deployment, see [`DEPLOYMENT.md`](DEPLOYMENT.md).

## Run the Web App

### React + FastAPI App

This is the recommended UI for the Live Job Finder.

Start the Python API:

```bash
pip install -r requirements.txt
uvicorn backend_api:app --reload
```

For JSearch, keep the RapidAPI key in the backend only. Create a local `.env`
file beside `backend_api.py`:

```bash
cp .env.example .env
```

Then set:

```text
RAPIDAPI_KEY=your_rapidapi_key_here
ADZUNA_APP_ID=your_adzuna_app_id_here
ADZUNA_APP_KEY=your_adzuna_app_key_here
```

`.env` is ignored by git and should not be committed.

In a second terminal, start the React app:

```bash
cd frontend
npm install
npm run dev
```

Then open the React URL shown by Vite, usually:

```text
http://localhost:5173
```

### Streamlit App

The older Streamlit interface is still available:

```bash
streamlit run app.py
```

The app supports:

- Pasting resume and job description text.
- Uploading `.txt`, `.pdf`, or `.docx` resumes.
- Finding live jobs from an uploaded resume in the `Live Job Finder` page.
- Loading examples from `resume_job_matching_dataset.csv`.
- Ranking one uploaded resume against many job descriptions.
- Uploading a jobs CSV and showing the most appropriate jobs with explanations.
- Scanning a folder of resumes, including:
  `/Users/jigyasaverma/Desktop/Resume_screening/pythonProject/merged_resumes`

## Live Job Finder

Streamlit automatically shows a separate `Live Job Finder` page from:

```text
pages/1_Live_Job_Finder.py
```

Use it when you want the system to:

- Upload a resume.
- Extract text from PDF, DOCX, or TXT.
- Infer suitable job search queries from resume skills.
- Fetch live jobs from JSearch via RapidAPI, Adzuna, or the Remotive public jobs API.
- Rank those jobs against the resume.
- Show ranked job cards, apply links, and explainable match reports for the best jobs.

For broader results from multiple job sources, choose `All Providers`. It fetches
from JSearch, Adzuna, SerpAPI Google Jobs, Jooble, Internshala via Apify, and
Remotive together, deduplicates overlapping jobs, and ranks the combined pool.
You can still choose `JSearch via RapidAPI`, `Adzuna`, `SerpAPI Google Jobs`,
`Jooble`, `Internshala via Apify`, or `Remotive` individually. The React app
does not ask for keys; the backend reads credentials from `.env` or your
environment.

```bash
export RAPIDAPI_KEY="your_rapidapi_key"
export ADZUNA_APP_ID="your_adzuna_app_id"
export ADZUNA_APP_KEY="your_adzuna_app_key"
export SERPAPI_KEY="your_serpapi_key"
export JOOBLE_API_KEY="your_jooble_api_key"
export APIFY_API_TOKEN="your_apify_api_token"
export APIFY_INTERNSHALA_ACTOR_ID="your_apify_actor_id"
uvicorn backend_api:app --reload
```

JSearch can return jobs from multiple publishers, depending on the API response.
Adzuna is a second credential-based provider. SerpAPI adds Google Jobs results,
which is useful for India-focused searches. Jooble adds another broad global job
aggregator. Internshala can be fetched through an Apify actor if you provide
`APIFY_API_TOKEN` and `APIFY_INTERNSHALA_ACTOR_ID`. Remotive remains available as
a no-key fallback, but it only returns Remotive jobs. In `All Providers` mode,
one provider can fail without breaking the whole search; the UI shows provider
warnings and still ranks jobs returned by the other sources.

The code also checks whether these API-key providers are configured:

- `RAPIDAPI_KEY` for RapidAPI JSearch.
- `ADZUNA_APP_ID` and `ADZUNA_APP_KEY` for Adzuna.
- `SERPAPI_KEY` for SerpAPI Google Jobs.
- `JOOBLE_API_KEY` for Jooble.
- `APIFY_API_TOKEN` and `APIFY_INTERNSHALA_ACTOR_ID` for Internshala via Apify.

Those providers can be added later without changing the resume matching logic.

## Job Recommendations

Open the `Job Recommendations` tab to upload one resume and compare it against
many jobs. You can use the included `resume_job_matching_dataset.csv` or upload a
CSV exported from job websites.

The jobs CSV should include a description column. Accepted names include:

- `job_description`
- `Job Description`
- `description`
- `jd`
- `details`
- `posting`

Optional columns improve the ranked output:

- Title: `title`, `job_title`, `role`, `position`
- Company: `company`, `company_name`, `organization`, `employer`
- URL: `url`, `link`, `job_url`, `posting_url`

The app ranks jobs by final score and shows:

- Semantic match score
- Skill match score
- Matched and missing skills
- ATS score
- Bias risk
- A full explainable report for each top job

## Train the Match Model

The analyzer can learn scoring from `resume_job_matching_dataset.csv`. Train it
with:

```bash
python3 train_match_model.py --dataset resume_job_matching_dataset.csv
```

This creates `trained_match_model.joblib`. After that, restart the backend or
Streamlit and new analyses will use the trained model automatically.

The training script:

- Uses `resume`, `job_description`, and `match_score` columns.
- Converts dataset scores like `1-5` into `0-100`.
- Builds explainable numeric features such as semantic score, skill score, ATS
  score, section quality, text overlap, and missing skill count.
- Uses a skill ontology graph for related-skill matching, such as
  `machine learning -> deep learning -> pytorch`.
- Extracts resume intelligence signals such as measurable project impact, tool
  depth, fresher/professional stage, and weighted section quality.
- Trains a `RandomForestRegressor` for pointwise match scoring.
- Trains a pairwise learning-to-rank classifier that learns which resume-job pair
  should rank higher when match scores differ.
- Prints regression metrics plus ranker pair accuracy so you can see model
  quality.

For a quick test run, train on fewer rows:

```bash
python3 train_match_model.py --dataset resume_job_matching_dataset.csv --limit 500
```

For larger or smaller ranker experiments:

```bash
python3 train_match_model.py --dataset resume_job_matching_dataset.csv --max-rank-pairs 75000 --min-rank-gap 10
```

## Evaluate Ranking Quality

Use ranking metrics instead of classification accuracy:

```bash
.venv/bin/python evaluate_ranking.py --dataset resume_job_matching_dataset.csv --limit 500 --group-size 20 --k 5,10
```

The evaluator reports:

- `NDCG@K` for graded ranking quality.
- `Precision@K` for how many top-K results are relevant.
- `Recall@K` for how many relevant results were recovered in top-K.
- Improvements for LTR and optional SBERT scoring versus the rule-based formula.

To include the transformer/SBERT formula comparison:

```bash
.venv/bin/python evaluate_ranking.py --dataset resume_job_matching_dataset.csv --include-sbert
```

The current CSV has one unique resume per row, so the evaluator uses shuffled
fixed-size groups as a benchmark proxy. If a future dataset includes `query_id`,
`resume_id`, `group_id`, or `candidate_group`, the evaluator will use those real
groups automatically.

## Run from CLI

```bash
python resume_analyzer.py \
  --resume-text "Experienced Python ML engineer..." \
  --job-text "Looking for an ML Engineer with Python, TensorFlow, and MLOps..."
```

You can also pass text files:

```bash
python resume_analyzer.py --resume-file resume.txt --job-file job.txt
```

## Notes

- By default, semantic scoring uses TF-IDF with skill synonym expansion. This
  avoids heavy transformer/torchvision dependencies while keeping the app easy to
  run.
- To opt into transformer embeddings, install `sentence-transformers` separately
  or use `pip install -r requirements-transformer.txt`, then run with
  `USE_TRANSFORMER_EMBEDDINGS=true`.
- Skill scoring is context-aware: applied evidence like "built/deployed" and
  measurable impact scores higher than weak evidence like "learned/familiar".
- Related skills are matched through an ontology graph, so adjacent tools and
  concepts can count as partial coverage instead of pure missing skills.
- Section quality is weighted by profile stage: freshers get more weight on
  projects/skills/education, while professionals get more weight on experience.
- Resume intelligence extracts impact statements like "improved accuracy by 20%"
  and tool depth labels such as learned, used, built, deployed, and optimized.
- If `trained_match_model.joblib` exists, final scoring uses the trained model.
  If not, it uses the explainable formula fallback.
- The bias check anonymizes gender-identifiable terms, rescoring the resume after
  anonymization and reporting any score difference.
