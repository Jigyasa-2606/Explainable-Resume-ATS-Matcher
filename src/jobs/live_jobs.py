from typing import Dict, List, Set

import requests

from src.features.build_phase2_features import SKILL_PHRASES, _skills_in_text


def _top_skill_query(text: str, max_skills: int = 3) -> str:
    skills = sorted(_skills_in_text((text or "").lower(), SKILL_PHRASES))
    if not skills:
        return "software engineer"
    return " ".join(skills[:max_skills])


def _fetch_remotive(query: str, limit: int = 8) -> List[Dict[str, str]]:
    url = "https://remotive.com/api/remote-jobs"
    resp = requests.get(url, params={"search": query}, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    jobs = payload.get("jobs", [])

    out: List[Dict[str, str]] = []
    for job in jobs[:limit]:
        out.append(
            {
                "source": "Remotive",
                "title": job.get("title", ""),
                "company": job.get("company_name", ""),
                "location": job.get("candidate_required_location", "Remote"),
                "url": job.get("url", ""),
            }
        )
    return out


def _fetch_arbeitnow(query: str, limit: int = 8) -> List[Dict[str, str]]:
    url = "https://www.arbeitnow.com/api/job-board-api"
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    jobs = payload.get("data", [])

    query_tokens: Set[str] = {tok for tok in query.lower().split() if tok}
    out: List[Dict[str, str]] = []
    fallback: List[Dict[str, str]] = []
    for job in jobs:
        title = (job.get("title") or "").lower()
        tags = " ".join(job.get("tags") or []).lower()
        text = f"{title} {tags}"
        row = (
            {
                "source": "Arbeitnow",
                "title": job.get("title", ""),
                "company": job.get("company_name", ""),
                "location": job.get("location", "N/A"),
                "url": job.get("url", ""),
            }
        )
        fallback.append(row)
        if not query_tokens:
            out.append(row)
        else:
            # Keep jobs with at least one query token match.
            hit_count = sum(1 for token in query_tokens if token in text)
            if hit_count >= 1:
                out.append(row)
        if len(out) >= limit:
            return out

    # Fallback if filtering was still too narrow.
    return fallback[:limit]


def fetch_live_jobs(resume_text: str, job_description: str, per_source_limit: int = 8) -> Dict[str, object]:
    search_query = _top_skill_query(f"{resume_text} {job_description}")
    jobs: List[Dict[str, str]] = []
    errors: List[str] = []

    try:
        jobs.extend(_fetch_remotive(search_query, limit=per_source_limit))
    except Exception as exc:  # pragma: no cover
        errors.append(f"Remotive fetch failed: {exc}")

    try:
        jobs.extend(_fetch_arbeitnow(search_query, limit=per_source_limit))
    except Exception as exc:  # pragma: no cover
        errors.append(f"Arbeitnow fetch failed: {exc}")

    # Deduplicate by URL.
    seen = set()
    deduped = []
    for job in jobs:
        url = job.get("url", "")
        if not url or url in seen:
            continue
        seen.add(url)
        deduped.append(job)

    if not deduped:
        # One more broad fallback query.
        try:
            deduped = _fetch_remotive("software engineer", limit=per_source_limit)
        except Exception:
            pass

    return {"query": search_query, "jobs": deduped, "errors": errors}
