from __future__ import annotations

import json
import re
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]

from backend_settings import load_local_env
from file_text import extract_text_from_upload
from job_sources import (
    JobSourceError,
    LiveJob,
    fetch_adzuna_jobs,
    fetch_jsearch_jobs,
    fetch_jooble_jobs,
    fetch_remotive_jobs,
    fetch_serpapi_jobs,
    live_jobs_to_postings,
)
from resume_analyzer import SkillMatch, infer_resume_job_queries, rank_jobs_for_resume, skill_graph_explanations


load_local_env()

app = FastAPI(title="Resume Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"^http://(localhost|127\.0\.0\.1):517[0-9]$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/live-jobs")
async def live_jobs(
    resume_file: UploadFile | None = File(default=None),
    resume_text: str = Form(default=""),
    query: str = Form(default=""),
    selected_roles: str = Form(default=""),
    provider: str = Form(default="jsearch"),
    country: str = Form(default="in"),
    date_posted: str = Form(default="month"),
    opportunity_type: str = Form(default="jobs"),
    result_limit: int = Form(default=25),
    top_n: int = Form(default=10),
) -> dict[str, Any]:
    resolved_resume_text = resume_text
    if resume_file is not None:
        content = await resume_file.read()
        extracted = extract_text_from_upload(resume_file.filename or "", content)
        if extracted.strip():
            resolved_resume_text = extracted

    if not resolved_resume_text.strip():
        raise HTTPException(status_code=400, detail="Upload or paste a resume first.")

    suggested_queries = infer_resume_job_queries(resolved_resume_text)
    role_queries = _parse_selected_roles(selected_roles)
    search_query = query.strip() or _combine_role_queries(role_queries) or suggested_queries[0]
    search_query = _apply_opportunity_type(search_query, opportunity_type)
    if not search_query:
        raise HTTPException(status_code=400, detail="A job search query is required.")

    live_job_results, provider_warnings = _fetch_jobs_for_provider(
        provider=provider,
        query=search_query,
        limit=result_limit,
        country=country,
        date_posted=date_posted,
    )

    live_job_results = _filter_opportunity_type(live_job_results, opportunity_type)
    ranked = rank_jobs_for_resume(
        resolved_resume_text,
        live_jobs_to_postings(live_job_results),
        limit=top_n,
    )
    live_job_by_index = {index: job for index, job in enumerate(live_job_results, start=1)}

    return {
        "resume_preview": resolved_resume_text[:3000],
        "suggested_queries": suggested_queries,
        "selected_roles": role_queries,
        "query_used": search_query,
        "provider": provider,
        "provider_warnings": provider_warnings,
        "opportunity_type": opportunity_type,
        "jobs_fetched": len(live_job_results),
        "jobs": [_serialize_ranked_job(item, live_job_by_index.get(int(item["source_index"]))) for item in ranked],
    }


@app.post("/api/infer-queries")
async def infer_queries(
    resume_file: UploadFile | None = File(default=None),
    resume_text: str = Form(default=""),
) -> dict[str, Any]:
    resolved_resume_text = resume_text
    if resume_file is not None:
        content = await resume_file.read()
        extracted = extract_text_from_upload(resume_file.filename or "", content)
        if extracted.strip():
            resolved_resume_text = extracted

    if not resolved_resume_text.strip():
        raise HTTPException(status_code=400, detail="Upload or paste a resume first.")

    return {
        "resume_preview": resolved_resume_text[:3000],
        "suggested_queries": infer_resume_job_queries(resolved_resume_text),
    }


def _serialize_ranked_job(item: dict[str, Any], live_job: Any) -> dict[str, Any]:
    analysis = item["analysis"]
    skill_match: SkillMatch = analysis["skill_match"]
    sections = analysis["sections"]
    bias = analysis["bias"]
    resume_intelligence = analysis["resume_intelligence"]

    return {
        "rank": item["rank"],
        "title": item["title"],
        "company": item["company"],
        "url": item["url"],
        "location": live_job.location if live_job else "",
        "source": live_job.source if live_job else "",
        "job_type": live_job.job_type if live_job else "",
        "publication_date": live_job.publication_date if live_job else "",
        "final_score": item["final_score"],
        "base_final_score": item.get("base_final_score"),
        "learning_to_rank_score": item.get("learning_to_rank_score"),
        "semantic_score": item["semantic_score"],
        "skill_score": item["skill_score"],
        "ats_score": item["ats_score"],
        "bias_risk": item["bias_risk"],
        "experience_fit": item.get("experience_fit", {}),
        "resume_intelligence": resume_intelligence,
        "scoring_method": analysis["scoring_method"],
        "summary": item["summary"],
        "matched_skills": item["matched_skills"],
        "missing_skills": item["missing_skills"],
        "partial_skills": item["partial_skills"],
        "skill_graph_matches": skill_match.graph_matches,
        "skill_graph_explanations": skill_graph_explanations(skill_match),
        "skill_evidence": skill_match.evidence,
        "strong_evidence_skills": [
            skill for skill, label in skill_match.evidence.items() if label == "strong"
        ],
        "weak_evidence_skills": [
            skill for skill, label in skill_match.evidence.items() if label == "weak"
        ],
        "analysis": {
            "final_score": analysis["final_score"],
            "base_final_score": analysis.get("base_final_score"),
            "learning_to_rank_score": analysis.get("learning_to_rank_score"),
            "semantic_score": analysis["semantic_score"],
            "skill_score": skill_match.score,
            "ats_score": analysis["ats_score"],
            "bias_risk": bias["risk_level"],
            "scoring_method": analysis["scoring_method"],
            "overall_explanation": analysis["overall_explanation"],
            "sections": sections,
            "resume_intelligence": resume_intelligence,
            "ats_suggestions": analysis["ats_suggestions"],
            "improvements": analysis["improvements"],
            "skill_evidence": skill_match.evidence,
            "skill_graph_matches": skill_match.graph_matches,
            "skill_graph_explanations": skill_graph_explanations(skill_match),
        },
    }


def _fetch_jobs_for_provider(
    provider: str,
    query: str,
    limit: int,
    country: str,
    date_posted: str,
) -> tuple[list[LiveJob], list[str]]:
    normalized_provider = provider.lower().strip()
    if normalized_provider == "all":
        return _fetch_all_providers(query, limit, country, date_posted)

    try:
        if normalized_provider == "remotive":
            return fetch_remotive_jobs(query, limit=limit), []
        if normalized_provider == "adzuna":
            return fetch_adzuna_jobs(query, limit=limit, country=country), []
        if normalized_provider == "serpapi":
            return fetch_serpapi_jobs(query, limit=limit, country=country), []
        if normalized_provider == "jooble":
            return fetch_jooble_jobs(query, limit=limit, country=country), []
        return fetch_jsearch_jobs(query, limit=limit, country=country, date_posted=date_posted), []
    except JobSourceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


def _fetch_all_providers(
    query: str,
    limit: int,
    country: str,
    date_posted: str,
) -> tuple[list[LiveJob], list[str]]:
    provider_calls = [
        ("JSearch", lambda: fetch_jsearch_jobs(query, limit=limit, country=country, date_posted=date_posted)),
        ("Adzuna", lambda: fetch_adzuna_jobs(query, limit=limit, country=country)),
        ("SerpAPI", lambda: fetch_serpapi_jobs(query, limit=limit, country=country)),
        ("Jooble", lambda: fetch_jooble_jobs(query, limit=limit, country=country)),
        ("Remotive", lambda: fetch_remotive_jobs(query, limit=limit)),
    ]
    jobs: list[LiveJob] = []
    warnings: list[str] = []

    for source_name, fetcher in provider_calls:
        try:
            source_jobs = fetcher()
            jobs.extend(source_jobs)
            if not source_jobs:
                warnings.append(f"{source_name} returned no jobs.")
        except JobSourceError as exc:
            warnings.append(f"{source_name}: {exc}")

    deduped_jobs = _deduplicate_live_jobs(jobs)
    if not deduped_jobs and warnings:
        raise HTTPException(status_code=502, detail="No providers returned jobs. " + " ".join(warnings))

    return deduped_jobs, warnings


def _deduplicate_live_jobs(jobs: list[LiveJob]) -> list[LiveJob]:
    seen = set()
    deduped = []
    for job in jobs:
        key = _job_dedup_key(job)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(job)
    return deduped


def _job_dedup_key(job: LiveJob) -> str:
    if job.url:
        normalized_url = re.sub(r"[?#].*$", "", job.url.strip().lower()).rstrip("/")
        if normalized_url:
            return f"url:{normalized_url}"

    pieces = [job.title, job.company, job.location]
    normalized = "|".join(re.sub(r"\s+", " ", piece.lower()).strip() for piece in pieces if piece)
    return f"text:{normalized}" if normalized else f"raw:{job.title}:{job.company}:{job.description[:80]}"


def _apply_opportunity_type(query: str, opportunity_type: str) -> str:
    normalized_type = opportunity_type.lower()
    normalized_query = query.lower()
    if normalized_type == "internships" and "intern" not in normalized_query:
        return f"{query} internship"
    return query


def _parse_selected_roles(selected_roles: str) -> list[str]:
    if not selected_roles.strip():
        return []
    try:
        parsed = json.loads(selected_roles)
    except json.JSONDecodeError:
        parsed = [item.strip() for item in selected_roles.split(",")]
    if not isinstance(parsed, list):
        return []
    return [str(item).strip() for item in parsed if str(item).strip()]


def _combine_role_queries(roles: list[str]) -> str:
    if not roles:
        return ""
    return " OR ".join(roles[:5])


def _filter_opportunity_type(live_jobs: list[Any], opportunity_type: str) -> list[Any]:
    if opportunity_type.lower() != "internships":
        return live_jobs

    internship_terms = ("intern", "internship", "trainee", "student")
    internship_jobs = [
        job
        for job in live_jobs
        if any(
            term in " ".join([job.title, job.description, job.job_type, job.category]).lower()
            for term in internship_terms
        )
    ]
    return internship_jobs or live_jobs
