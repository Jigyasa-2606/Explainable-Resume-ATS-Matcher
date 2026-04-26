from __future__ import annotations

import html
import math
import os
import re
from dataclasses import dataclass

from resume_analyzer import JobPosting


REMOTIVE_API_URL = "https://remotive.com/api/remote-jobs"
JSEARCH_API_URL = "https://jsearch.p.rapidapi.com/search"
JSEARCH_API_HOST = "jsearch.p.rapidapi.com"
ADZUNA_API_URL_TEMPLATE = "https://api.adzuna.com/v1/api/jobs/{country}/search/1"
SERPAPI_GOOGLE_JOBS_URL = "https://serpapi.com/search.json"
JOOBLE_API_URL_TEMPLATE = "https://jooble.org/api/{api_key}"

COUNTRY_LOCATIONS = {
    "in": "India",
    "us": "United States",
    "gb": "United Kingdom",
    "ca": "Canada",
    "au": "Australia",
    "sg": "Singapore",
}


@dataclass(frozen=True)
class LiveJob:
    title: str
    company: str
    description: str
    url: str
    location: str = ""
    source: str = "Remotive"
    category: str = ""
    job_type: str = ""
    publication_date: str = ""

    def to_job_posting(self) -> JobPosting:
        metadata = " ".join(
            part
            for part in [
                self.title,
                self.company,
                self.location,
                self.category,
                self.job_type,
                self.description,
            ]
            if part
        )
        return JobPosting(
            title=self.title,
            company=self.company,
            url=self.url,
            description=metadata,
        )


class JobSourceError(RuntimeError):
    pass


def fetch_remotive_jobs(query: str, limit: int = 25) -> list[LiveJob]:
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as exc:
        raise JobSourceError("Install dependencies first with: pip install -r requirements.txt") from exc

    params = {"search": query, "limit": limit}
    response = requests.get(REMOTIVE_API_URL, params=params, timeout=20)
    if response.status_code != 200:
        raise JobSourceError(f"Remotive returned HTTP {response.status_code}.")

    payload = response.json()
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        raise JobSourceError("Unexpected Remotive response format.")

    return [normalize_remotive_job(job) for job in jobs[:limit]]


def fetch_jsearch_jobs(
    query: str,
    rapidapi_key: str | None = None,
    limit: int = 25,
    country: str = "in",
    date_posted: str = "month",
) -> list[LiveJob]:
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as exc:
        raise JobSourceError("Install dependencies first with: pip install -r requirements.txt") from exc

    api_key = rapidapi_key or os.getenv("RAPIDAPI_KEY", "")
    if not api_key.strip():
        raise JobSourceError("Provide a RapidAPI key for JSearch.")

    page_size = max(1, min(limit, 100))
    # JSearch commonly returns around 10 jobs per page, so request enough pages
    # to satisfy the UI fetch limit and then cap locally.
    num_pages = max(1, min(10, math.ceil(page_size / 10)))
    params = {
        "query": query,
        "page": "1",
        "num_pages": str(num_pages),
        "country": country.lower(),
        "date_posted": date_posted,
    }
    headers = {
        "X-RapidAPI-Key": api_key.strip(),
        "X-RapidAPI-Host": JSEARCH_API_HOST,
    }
    response = requests.get(JSEARCH_API_URL, headers=headers, params=params, timeout=25)
    if response.status_code in {401, 403}:
        raise JobSourceError("JSearch rejected the RapidAPI key or subscription.")
    if response.status_code != 200:
        raise JobSourceError(f"JSearch returned HTTP {response.status_code}: {response.text[:200]}")

    payload = response.json()
    jobs = payload.get("data", [])
    if not isinstance(jobs, list):
        raise JobSourceError("Unexpected JSearch response format.")

    return [normalize_jsearch_job(job) for job in jobs[:page_size]]


def fetch_adzuna_jobs(
    query: str,
    limit: int = 25,
    country: str = "in",
) -> list[LiveJob]:
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as exc:
        raise JobSourceError("Install dependencies first with: pip install -r requirements.txt") from exc

    app_id = os.getenv("ADZUNA_APP_ID", "")
    app_key = os.getenv("ADZUNA_APP_KEY", "")
    if not app_id.strip() or not app_key.strip():
        raise JobSourceError("Provide ADZUNA_APP_ID and ADZUNA_APP_KEY in backend .env.")

    page_size = max(1, min(limit, 50))
    url = ADZUNA_API_URL_TEMPLATE.format(country=country.lower())
    params = {
        "app_id": app_id.strip(),
        "app_key": app_key.strip(),
        "what": query,
        "results_per_page": page_size,
        "content-type": "application/json",
    }
    response = requests.get(url, params=params, timeout=25)
    if response.status_code in {401, 403}:
        raise JobSourceError("Adzuna rejected the app credentials.")
    if response.status_code != 200:
        raise JobSourceError(f"Adzuna returned HTTP {response.status_code}: {response.text[:200]}")

    payload = response.json()
    jobs = payload.get("results", [])
    if not isinstance(jobs, list):
        raise JobSourceError("Unexpected Adzuna response format.")

    return [normalize_adzuna_job(job) for job in jobs[:page_size]]


def fetch_serpapi_jobs(
    query: str,
    limit: int = 25,
    country: str = "in",
) -> list[LiveJob]:
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as exc:
        raise JobSourceError("Install dependencies first with: pip install -r requirements.txt") from exc

    api_key = os.getenv("SERPAPI_KEY", "")
    if not api_key.strip():
        raise JobSourceError("Provide SERPAPI_KEY in backend .env.")

    page_size = max(1, min(limit, 100))
    params = {
        "engine": "google_jobs",
        "q": query,
        "api_key": api_key.strip(),
        "gl": country.lower(),
        "hl": "en",
    }
    response = requests.get(SERPAPI_GOOGLE_JOBS_URL, params=params, timeout=25)
    if response.status_code in {401, 403}:
        raise JobSourceError("SerpAPI rejected the API key.")
    if response.status_code != 200:
        raise JobSourceError(f"SerpAPI returned HTTP {response.status_code}: {response.text[:200]}")

    payload = response.json()
    error_message = payload.get("error")
    if error_message:
        raise JobSourceError(f"SerpAPI error: {error_message}")
    jobs = payload.get("jobs_results", [])
    if not isinstance(jobs, list):
        raise JobSourceError("Unexpected SerpAPI response format.")

    return [normalize_serpapi_job(job) for job in jobs[:page_size]]


def fetch_jooble_jobs(
    query: str,
    limit: int = 25,
    country: str = "in",
) -> list[LiveJob]:
    try:
        import requests  # type: ignore[import-not-found]
    except ImportError as exc:
        raise JobSourceError("Install dependencies first with: pip install -r requirements.txt") from exc

    api_key = os.getenv("JOOBLE_API_KEY", "")
    if not api_key.strip():
        raise JobSourceError("Provide JOOBLE_API_KEY in backend .env.")

    page_size = max(1, min(limit, 100))
    url = JOOBLE_API_URL_TEMPLATE.format(api_key=api_key.strip())
    payload = {
        "keywords": query,
        "location": COUNTRY_LOCATIONS.get(country.lower(), country.upper()),
        "page": 1,
    }
    response = requests.post(url, json=payload, timeout=25)
    if response.status_code in {401, 403}:
        raise JobSourceError("Jooble rejected the API key.")
    if response.status_code != 200:
        raise JobSourceError(f"Jooble returned HTTP {response.status_code}: {response.text[:200]}")

    data = response.json()
    jobs = data.get("jobs", [])
    if not isinstance(jobs, list):
        raise JobSourceError("Unexpected Jooble response format.")

    return [normalize_jooble_job(job) for job in jobs[:page_size]]


def normalize_remotive_job(job: dict[str, object]) -> LiveJob:
    return LiveJob(
        title=str(job.get("title", "")).strip(),
        company=str(job.get("company_name", "")).strip(),
        description=clean_html(str(job.get("description", ""))),
        url=str(job.get("url", "")).strip(),
        location=str(job.get("candidate_required_location", "")).strip(),
        category=str(job.get("category", "")).strip(),
        job_type=str(job.get("job_type", "")).strip(),
        publication_date=str(job.get("publication_date", "")).strip(),
    )


def normalize_adzuna_job(job: dict[str, object]) -> LiveJob:
    company = job.get("company")
    location = job.get("location")
    category = job.get("category")

    company_name = company.get("display_name", "") if isinstance(company, dict) else ""
    location_name = location.get("display_name", "") if isinstance(location, dict) else ""
    category_name = category.get("label", "") if isinstance(category, dict) else ""

    return LiveJob(
        title=str(job.get("title", "")).strip(),
        company=str(company_name).strip(),
        description=clean_html(str(job.get("description", ""))),
        url=str(job.get("redirect_url", "")).strip(),
        location=str(location_name).strip(),
        source="Adzuna",
        category=str(category_name).strip(),
        job_type=str(job.get("contract_type", "") or job.get("contract_time", "")).strip(),
        publication_date=str(job.get("created", "")).strip(),
    )


def normalize_jooble_job(job: dict[str, object]) -> LiveJob:
    snippet = str(job.get("snippet", "") or job.get("description", "")).strip()
    salary = str(job.get("salary", "")).strip()
    source = str(job.get("source", "")).strip()
    job_type = str(job.get("type", "")).strip()
    description_parts = [snippet]
    if salary:
        description_parts.append(f"Salary: {salary}")
    if source:
        description_parts.append(f"Source: {source}")

    return LiveJob(
        title=str(job.get("title", "")).strip(),
        company=str(job.get("company", "")).strip(),
        description=clean_html(" ".join(part for part in description_parts if part)),
        url=str(job.get("link", "")).strip(),
        location=str(job.get("location", "")).strip(),
        source="Jooble",
        category=str(source or "Jooble").strip(),
        job_type=job_type,
        publication_date=str(job.get("updated", "")).strip(),
    )


def normalize_serpapi_job(job: dict[str, object]) -> LiveJob:
    apply_link = ""
    apply_options = job.get("apply_options")
    if isinstance(apply_options, list) and apply_options:
        first_option = apply_options[0]
        if isinstance(first_option, dict):
            apply_link = str(first_option.get("link", "")).strip()
    if not apply_link:
        related_links = job.get("related_links")
        if isinstance(related_links, list) and related_links:
            first_link = related_links[0]
            if isinstance(first_link, dict):
                apply_link = str(first_link.get("link", "")).strip()
    if not apply_link:
        apply_link = str(job.get("share_link", "")).strip()

    extensions = job.get("detected_extensions")
    schedule_type = ""
    posted_at = ""
    if isinstance(extensions, dict):
        schedule_type = str(extensions.get("schedule_type", "")).strip()
        posted_at = str(extensions.get("posted_at", "")).strip()

    description_parts = [
        str(job.get("description", "")).strip(),
        " ".join(str(item) for item in job.get("job_highlights", []) if isinstance(item, str)),
    ]

    return LiveJob(
        title=str(job.get("title", "")).strip(),
        company=str(job.get("company_name", "")).strip(),
        description=clean_html(" ".join(part for part in description_parts if part)),
        url=apply_link,
        location=str(job.get("location", "")).strip(),
        source="SerpAPI Google Jobs",
        category="Google Jobs",
        job_type=schedule_type,
        publication_date=posted_at,
    )


def normalize_jsearch_job(job: dict[str, object]) -> LiveJob:
    apply_link = str(job.get("job_apply_link") or job.get("job_google_link") or "").strip()
    location_parts = [
        str(job.get("job_city") or "").strip(),
        str(job.get("job_state") or "").strip(),
        str(job.get("job_country") or "").strip(),
    ]
    location = ", ".join(part for part in location_parts if part)
    description = clean_html(str(job.get("job_description", "")))
    employment_types = job.get("job_employment_types") or []
    if isinstance(employment_types, list):
        job_type = ", ".join(str(item) for item in employment_types)
    else:
        job_type = str(employment_types)

    return LiveJob(
        title=str(job.get("job_title", "")).strip(),
        company=str(job.get("employer_name", "")).strip(),
        description=description,
        url=apply_link,
        location=location,
        source=str(job.get("job_publisher") or "JSearch").strip(),
        category=str(job.get("job_occupational_categories") or "").strip(),
        job_type=job_type,
        publication_date=str(job.get("job_posted_at_datetime_utc") or "").strip(),
    )


def clean_html(value: str) -> str:
    text = re.sub(r"<(br|p|li|div|h[1-6])[^>]*>", " ", value, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def configured_job_sources() -> dict[str, bool]:
    return {
        "remotive": True,
        "rapidapi_jsearch": bool(os.getenv("RAPIDAPI_KEY")),
        "adzuna": bool(os.getenv("ADZUNA_APP_ID") and os.getenv("ADZUNA_APP_KEY")),
        "serpapi": bool(os.getenv("SERPAPI_KEY")),
        "jooble": bool(os.getenv("JOOBLE_API_KEY")),
    }


def live_jobs_to_postings(jobs: list[LiveJob]) -> list[JobPosting]:
    return [job.to_job_posting() for job in jobs if job.description.strip() or job.title.strip()]
