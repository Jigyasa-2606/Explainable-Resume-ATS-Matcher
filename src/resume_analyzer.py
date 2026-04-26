"""Explainable resume-to-job-description analyzer.

The module is intentionally dependency-light at import time. If
sentence-transformers is installed, semantic similarity uses embeddings. If not,
it falls back to TF-IDF over normalized text so the app still works locally.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


TRAINED_MODEL_PATH = Path(__file__).with_name("trained_match_model.joblib")
SECTION_SCORE_POINTS = {"Good": 100, "Average": 60, "Poor": 20}
USE_TRANSFORMER_EMBEDDINGS = os.getenv("USE_TRANSFORMER_EMBEDDINGS", "").lower() in {"1", "true", "yes"}

SECTION_HEADINGS = {
    "education": ["education", "academic background", "academics", "qualification"],
    "experience": [
        "experience",
        "work experience",
        "professional experience",
        "employment",
        "work history",
    ],
    "projects": ["projects", "project experience", "academic projects", "portfolio"],
    "skills": ["skills", "technical skills", "core competencies", "technologies"],
}

ACTION_VERBS = {
    "achieved",
    "analyzed",
    "automated",
    "built",
    "collaborated",
    "created",
    "delivered",
    "designed",
    "developed",
    "implemented",
    "improved",
    "led",
    "managed",
    "optimized",
    "reduced",
    "reported",
    "shipped",
    "solved",
    "trained",
}

STRONG_EVIDENCE_TERMS = ACTION_VERBS | {
    "architected",
    "deployed",
    "engineered",
    "evaluated",
    "launched",
    "measured",
    "modeled",
    "predicted",
    "produced",
    "validated",
}

WEAK_EVIDENCE_TERMS = {
    "aware",
    "basic",
    "beginner",
    "course",
    "coursework",
    "exposure",
    "familiar",
    "introductory",
    "learned",
    "learning",
    "studied",
    "understanding",
}

TOOL_DEPTH_LABELS = ["learned", "used", "built", "deployed", "optimized"]
TOOL_DEPTH_POINTS = {
    "learned": 20,
    "used": 45,
    "built": 70,
    "deployed": 85,
    "optimized": 95,
}
TOOL_DEPTH_TERMS = {
    "optimized": {"optimized", "improved", "reduced", "accelerated", "increased", "decreased", "boosted"},
    "deployed": {"deployed", "launched", "shipped", "production", "released", "hosted"},
    "built": {"built", "developed", "implemented", "designed", "created", "trained", "engineered"},
    "used": {"used", "applied", "worked", "analyzed", "created", "performed", "leveraged"},
    "learned": WEAK_EVIDENCE_TERMS,
}

FRESHER_SECTION_WEIGHTS = {
    "education": 0.25,
    "experience": 0.15,
    "projects": 0.35,
    "skills": 0.25,
}
PROFESSIONAL_SECTION_WEIGHTS = {
    "education": 0.15,
    "experience": 0.40,
    "projects": 0.25,
    "skills": 0.20,
}

SKILL_ALIASES = {
    "agile": ["agile", "scrum"],
    "artificial intelligence": ["artificial intelligence", "ai"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "backend development": ["backend development", "backend engineering", "backend"],
    "cnn": ["cnn", "convolutional neural network", "convolutional neural networks"],
    "computer vision": ["computer vision", "cv"],
    "cloud": ["cloud", "cloud computing"],
    "data analysis": ["data analysis", "analytics"],
    "data cleaning": ["data cleaning", "data cleansing", "data preprocessing"],
    "data visualization": ["data visualization", "visualization", "dashboards"],
    "deep learning": ["deep learning", "neural networks", "dl"],
    "docker": ["docker", "containerization", "containers"],
    "excel": ["excel", "spreadsheets", "microsoft excel"],
    "git": ["git", "github", "version control"],
    "java": ["java"],
    "keras": ["keras"],
    "machine learning": ["machine learning", "ml"],
    "mlops": ["mlops", "model deployment", "model monitoring"],
    "nlp": ["nlp", "natural language processing"],
    "pandas": ["pandas"],
    "power bi": ["power bi", "powerbi"],
    "pytorch": ["pytorch", "py torch"],
    "python": ["python"],
    "reporting": ["reporting", "business reporting", "reports"],
    "rest apis": ["rest apis", "rest api", "restful api", "api development"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "spring boot": ["spring boot", "spring"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "relational databases"],
    "statistics": ["statistics", "statistical analysis", "stats"],
    "system design": ["system design", "architecture", "software architecture"],
    "tableau": ["tableau"],
    "tensorflow": ["tensorflow", "tf"],
}

SKILL_GRAPH = {
    "artificial intelligence": {"machine learning", "nlp", "computer vision"},
    "machine learning": {"deep learning", "scikit-learn", "statistics", "pandas", "mlops"},
    "deep learning": {"cnn", "tensorflow", "keras", "pytorch", "computer vision", "nlp"},
    "computer vision": {"cnn", "tensorflow", "keras", "pytorch"},
    "data analysis": {"sql", "excel", "pandas", "statistics", "data cleaning", "data visualization"},
    "data visualization": {"power bi", "tableau", "reporting"},
    "backend development": {"java", "spring boot", "rest apis", "system design", "docker", "git"},
    "cloud": {"aws", "azure", "docker", "mlops"},
}

RELATED_SKILLS = {
    "machine learning": {"deep learning", "scikit-learn", "tensorflow", "keras", "mlops"},
    "deep learning": {"machine learning", "tensorflow", "keras", "computer vision", "nlp"},
    "nlp": {"machine learning", "deep learning", "python"},
    "computer vision": {"machine learning", "deep learning", "tensorflow", "keras"},
    "data visualization": {"power bi", "tableau", "reporting"},
    "reporting": {"data visualization", "power bi", "tableau", "excel"},
    "sql": {"data cleaning", "reporting", "data visualization"},
    "python": {"pandas", "machine learning", "data cleaning"},
    "rest apis": {"spring boot", "java", "system design"},
    "spring boot": {"java", "rest apis"},
    "docker": {"mlops", "system design"},
}

ROLE_QUERY_RULES = [
    ("machine learning engineer", {"python", "machine learning", "tensorflow", "keras", "mlops"}),
    ("data scientist", {"python", "machine learning", "statistics", "pandas", "nlp"}),
    ("data analyst", {"sql", "excel", "power bi", "tableau", "reporting", "data cleaning"}),
    ("software engineer", {"java", "spring boot", "rest apis", "git", "docker", "system design"}),
    ("python developer", {"python", "rest apis", "git", "docker"}),
    ("business intelligence analyst", {"power bi", "tableau", "sql", "reporting", "data visualization"}),
]

GENDERED_TERMS = {
    "he",
    "him",
    "his",
    "she",
    "her",
    "hers",
    "male",
    "female",
    "man",
    "woman",
    "mr",
    "mrs",
    "ms",
    "miss",
    "father",
    "mother",
    "husband",
    "wife",
    "boy",
    "girl",
}

GENDERED_JOB_WORDS = {
    "aggressive",
    "assertive",
    "dominant",
    "ninja",
    "rockstar",
    "guru",
    "supportive",
    "nurturing",
    "empathetic",
    "compassionate",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


@dataclass(frozen=True)
class SkillMatch:
    matched: list[str]
    missing: list[str]
    partial: list[str]
    resume_skills: list[str]
    job_skills: list[str]
    score: float
    evidence: dict[str, str]
    evidence_score: float
    graph_matches: dict[str, list[str]]


@dataclass(frozen=True)
class JobPosting:
    description: str
    title: str = ""
    company: str = ""
    url: str = ""


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def anonymize_gender(text: str) -> str:
    text = re.sub(r"\b(mr|mrs|ms|miss)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?", "[NAME]", text)
    pattern = re.compile(r"\b(" + "|".join(sorted(GENDERED_TERMS, key=len, reverse=True)) + r")\b", re.I)
    return pattern.sub("[GENDERED_TERM]", text)


def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.lower()).replace(r"\ ", r"\s+")
    return re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])", re.I)


def extract_skills(text: str) -> list[str]:
    normalized = normalize_text(text)
    found = []
    for canonical, aliases in SKILL_ALIASES.items():
        if any(_alias_pattern(alias).search(normalized) for alias in aliases):
            found.append(canonical)
    return sorted(found)


@lru_cache(maxsize=1)
def skill_graph_adjacency() -> dict[str, set[str]]:
    adjacency: dict[str, set[str]] = {skill: set() for skill in SKILL_ALIASES}
    for parent, children in SKILL_GRAPH.items():
        adjacency.setdefault(parent, set())
        for child in children:
            adjacency.setdefault(child, set())
            adjacency[parent].add(child)
            adjacency[child].add(parent)

    for skill, related_skills in RELATED_SKILLS.items():
        adjacency.setdefault(skill, set())
        for related in related_skills:
            adjacency.setdefault(related, set())
            adjacency[skill].add(related)
            adjacency[related].add(skill)

    return adjacency


def skill_graph_path(source: str, target: str, max_depth: int = 2) -> list[str]:
    if source == target:
        return [source]

    adjacency = skill_graph_adjacency()
    queue: list[tuple[str, list[str]]] = [(source, [source])]
    visited = {source}

    while queue:
        current, path = queue.pop(0)
        if len(path) - 1 >= max_depth:
            continue
        for neighbor in sorted(adjacency.get(current, set())):
            if neighbor in visited:
                continue
            next_path = path + [neighbor]
            if neighbor == target:
                return next_path
            visited.add(neighbor)
            queue.append((neighbor, next_path))

    return []


def related_skills_from_graph(skill: str, max_depth: int = 2) -> set[str]:
    adjacency = skill_graph_adjacency()
    related = set()
    queue: list[tuple[str, int]] = [(skill, 0)]
    visited = {skill}

    while queue:
        current, depth = queue.pop(0)
        if depth >= max_depth:
            continue
        for neighbor in adjacency.get(current, set()):
            if neighbor in visited:
                continue
            visited.add(neighbor)
            related.add(neighbor)
            queue.append((neighbor, depth + 1))

    return related


def infer_resume_job_queries(resume_text: str, max_queries: int = 4) -> list[str]:
    resume_skills = set(extract_skills(resume_text))
    ontology_skills = set(resume_skills)
    for skill in resume_skills:
        ontology_skills.update(related_skills_from_graph(skill, max_depth=1))
    normalized = normalize_text(resume_text)
    scored_queries = []

    for role, role_skills in ROLE_QUERY_RULES:
        overlap = resume_skills & role_skills
        related_overlap = (ontology_skills & role_skills) - overlap
        score = (len(overlap) * 2) + len(related_overlap)
        if role in normalized:
            score += 3
        if score > 0:
            scored_queries.append((score, role))

    scored_queries.sort(key=lambda item: (-item[0], item[1]))
    queries = [role for _, role in scored_queries]

    if not queries and resume_skills:
        primary_skills = " ".join(sorted(resume_skills)[:3])
        queries.append(primary_skills)

    if not queries:
        queries.append("software developer")

    return queries[:max_queries]


def classify_skill_evidence(resume_text: str, skill: str) -> str:
    normalized_resume = normalize_text(resume_text)
    aliases = SKILL_ALIASES.get(skill, [skill])
    evidence_windows = skill_evidence_windows(normalized_resume, aliases)

    if not evidence_windows:
        return "missing"

    impact_pattern = re.compile(r"(\d+(?:\.\d+)?\s*%|\b\d+x\b|\b\d+\s*(?:users|records|rows|clients|projects|models)\b)")
    for window in evidence_windows:
        tokens = set(window.split())
        if tokens & STRONG_EVIDENCE_TERMS or impact_pattern.search(window):
            return "strong"

    for window in evidence_windows:
        tokens = set(window.split())
        if tokens & WEAK_EVIDENCE_TERMS:
            return "weak"

    return "mentioned"


def skill_evidence_windows(normalized_resume: str, aliases: list[str]) -> list[str]:
    evidence_windows = []

    for alias in aliases:
        for match in _alias_pattern(alias).finditer(normalized_resume):
            start = max(0, match.start() - 110)
            end = min(len(normalized_resume), match.end() + 110)
            evidence_windows.append(normalized_resume[start:end])

    return evidence_windows


def context_aware_skill_score(
    resume_text: str,
    job_skills: set[str],
    matched: list[str],
    partial: list[str],
    graph_matches: dict[str, list[str]] | None = None,
) -> tuple[float, dict[str, str]]:
    if not job_skills:
        return 0.0, {}

    evidence: dict[str, str] = {}
    total = 0.0
    for skill in sorted(job_skills):
        if skill in matched:
            label = classify_skill_evidence(resume_text, skill)
            evidence[skill] = label
            if label == "strong":
                total += 1.0
            elif label == "mentioned":
                total += 0.75
            elif label == "weak":
                total += 0.4
        elif skill in partial:
            evidence[skill] = "partial"
            graph_path = (graph_matches or {}).get(skill, [])
            total += 0.45 if graph_path and len(graph_path) <= 2 else 0.3
        else:
            evidence[skill] = "missing"

    return round((total / len(job_skills)) * 100, 2), evidence


def skill_graph_explanations(skill_match: SkillMatch) -> list[str]:
    explanations = []
    for target_skill, path in sorted(skill_match.graph_matches.items()):
        if len(path) >= 2:
            explanations.append(" -> ".join(path) + f" supports {target_skill}")
    return explanations


def split_resume_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [sentence.strip(" -*•\t") for sentence in sentences if sentence.strip(" -*•\t")]


def extract_project_impacts(resume_text: str, limit: int = 5) -> list[str]:
    metric_pattern = re.compile(
        r"(\d+(?:\.\d+)?\s*%|\b\d+(?:\.\d+)?x\b|\b\d+[kKmM]?\+?\s*"
        r"(?:users|records|rows|clients|customers|projects|models|requests|seconds|minutes|hours|days)\b)"
    )
    impact_verbs = {
        "achieved",
        "automated",
        "boosted",
        "decreased",
        "delivered",
        "improved",
        "increased",
        "optimized",
        "reduced",
        "saved",
        "scaled",
    }
    impacts = []
    for sentence in split_resume_sentences(resume_text):
        normalized = normalize_text(sentence)
        if metric_pattern.search(normalized) or (set(normalized.split()) & impact_verbs):
            if any(term in normalized for term in ["project", "built", "developed", "implemented", "model", "system", "dashboard"]):
                impacts.append(sentence)
        if len(impacts) >= limit:
            break
    return impacts


def classify_tool_depth(resume_text: str, skill: str) -> str:
    normalized_resume = normalize_text(resume_text)
    windows = skill_evidence_windows(normalized_resume, SKILL_ALIASES.get(skill, [skill]))
    if not windows:
        return "learned"

    best_label = "used"
    best_score = TOOL_DEPTH_POINTS[best_label]
    for window in windows:
        tokens = set(window.split())
        for label in TOOL_DEPTH_LABELS:
            if tokens & TOOL_DEPTH_TERMS[label] and TOOL_DEPTH_POINTS[label] > best_score:
                best_label = label
                best_score = TOOL_DEPTH_POINTS[label]

    return best_label


def extract_tool_depths(resume_text: str) -> dict[str, str]:
    return {
        skill: classify_tool_depth(resume_text, skill)
        for skill in extract_skills(resume_text)
    }


def classify_candidate_stage(resume_text: str) -> str:
    normalized = normalize_text(resume_text)
    resume_years = extract_years_of_experience(resume_text)
    fresher_terms = {"fresher", "student", "graduate", "entry level", "internship", "intern"}
    professional_terms = {"senior", "lead", "manager", "professional experience", "work experience"}

    if resume_years >= 2 or any(term in normalized for term in professional_terms):
        return "professional"
    if resume_years <= 1 or any(term in normalized for term in fresher_terms):
        return "fresher"
    return "professional"


def section_weights_for_stage(candidate_stage: str) -> dict[str, float]:
    if candidate_stage == "fresher":
        return FRESHER_SECTION_WEIGHTS
    return PROFESSIONAL_SECTION_WEIGHTS


def weighted_section_score(
    sections: dict[str, dict[str, str]],
    candidate_stage: str,
) -> float:
    weights = section_weights_for_stage(candidate_stage)
    return sum(SECTION_SCORE_POINTS[sections[section]["rating"]] * weight for section, weight in weights.items())


def analyze_resume_intelligence(
    resume_text: str,
    sections: dict[str, dict[str, str]] | None = None,
) -> dict[str, object]:
    section_data = sections if sections is not None else analyze_sections(resume_text)
    tool_depths = extract_tool_depths(resume_text)
    depth_values = [TOOL_DEPTH_POINTS[label] for label in tool_depths.values()]
    candidate_stage = classify_candidate_stage(resume_text)
    impacts = extract_project_impacts(resume_text)

    return {
        "candidate_stage": candidate_stage,
        "section_weights": section_weights_for_stage(candidate_stage),
        "weighted_section_score": round(weighted_section_score(section_data, candidate_stage), 2),
        "project_impacts": impacts,
        "impact_count": len(impacts),
        "tool_depths": tool_depths,
        "average_tool_depth": round(sum(depth_values) / max(len(depth_values), 1), 2),
        "advanced_tool_count": sum(1 for label in tool_depths.values() if label in {"built", "deployed", "optimized"}),
    }


def extract_years_of_experience(text: str) -> float:
    normalized = normalize_text(text)
    year_values: list[float] = []

    range_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:\+?\s*)?(?:years?|yrs?)"
    )
    single_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)")

    for match in range_pattern.finditer(normalized):
        year_values.append(float(match.group(2)))

    for match in single_pattern.finditer(normalized):
        year_values.append(float(match.group(1)))

    if any(term in normalized for term in ["fresher", "entry level", "no experience", "internship", "student"]):
        year_values.append(0.0)

    if not year_values:
        return 0.0
    return max(year_values)


def extract_required_experience_range(job_text: str) -> tuple[float | None, float | None]:
    normalized = normalize_text(job_text)
    range_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*(?:\+?\s*)?(?:years?|yrs?)"
    )
    single_pattern = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)")

    ranges = [(float(match.group(1)), float(match.group(2))) for match in range_pattern.finditer(normalized)]
    if ranges:
        min_years = min(item[0] for item in ranges)
        max_years = max(item[1] for item in ranges)
        return min_years, max_years

    singles = [float(match.group(1)) for match in single_pattern.finditer(normalized)]
    if singles:
        min_years = min(singles)
        return min_years, None

    if any(term in normalized for term in ["entry level", "fresher", "internship", "intern", "student"]):
        return 0.0, 1.0

    return None, None


def experience_fit(resume_text: str, job_text: str) -> dict[str, object]:
    resume_years = extract_years_of_experience(resume_text)
    required_min, required_max = extract_required_experience_range(job_text)

    penalty = 0
    warning = ""
    if required_min is not None and resume_years < required_min:
        gap = required_min - resume_years
        penalty = int(min(30, 10 + gap * 5))
        warning = (
            f"Experience mismatch: resume shows about {resume_years:g} years, "
            f"but the job asks for at least {required_min:g} years."
        )

    return {
        "resume_years": resume_years,
        "required_min_years": required_min,
        "required_max_years": required_max,
        "penalty": penalty,
        "warning": warning,
    }


def compare_skills(resume_text: str, job_text: str) -> SkillMatch:
    resume_skills = set(extract_skills(resume_text))
    job_skills = set(extract_skills(job_text))
    matched = sorted(job_skills & resume_skills)

    partial = []
    graph_matches: dict[str, list[str]] = {}
    for skill in sorted(job_skills - resume_skills):
        best_path: list[str] = []
        for resume_skill in sorted(resume_skills):
            path = skill_graph_path(resume_skill, skill, max_depth=2)
            if path and (not best_path or len(path) < len(best_path)):
                best_path = path
        if best_path:
            partial.append(skill)
            graph_matches[skill] = best_path

    missing = sorted(job_skills - set(matched) - set(partial))
    score, evidence = context_aware_skill_score(resume_text, job_skills, matched, partial, graph_matches)

    return SkillMatch(
        matched=matched,
        missing=missing,
        partial=partial,
        resume_skills=sorted(resume_skills),
        job_skills=sorted(job_skills),
        score=round(score, 2),
        evidence=evidence,
        evidence_score=round(score, 2),
        graph_matches=graph_matches,
    )


def _expand_for_similarity(text: str) -> str:
    expanded = [text]
    normalized = normalize_text(text)
    for canonical, aliases in SKILL_ALIASES.items():
        if any(_alias_pattern(alias).search(normalized) for alias in aliases):
            expanded.append(canonical)
            expanded.extend(aliases)
            expanded.extend(sorted(related_skills_from_graph(canonical, max_depth=1)))
    return " ".join(expanded)


@lru_cache(maxsize=1)
def _sentence_transformer_model() -> object:
    from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

    return SentenceTransformer("all-MiniLM-L6-v2")


def semantic_similarity(
    resume_text: str,
    job_text: str,
    use_transformer: bool = USE_TRANSFORMER_EMBEDDINGS,
) -> float:
    if not resume_text.strip() or not job_text.strip():
        return 0.0

    resume_expanded = _expand_for_similarity(resume_text)
    job_expanded = _expand_for_similarity(job_text)
    tfidf_score = _tfidf_similarity(resume_expanded, job_expanded)

    if use_transformer:
        try:
            from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-not-found]

            model = _sentence_transformer_model()
            embeddings = model.encode([resume_expanded, job_expanded])
            transformer_score = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            score = (0.7 * transformer_score) + (0.3 * tfidf_score)
        except Exception:
            score = tfidf_score
    else:
        score = tfidf_score

    return round(max(0.0, min(1.0, float(score))) * 100, 2)


def _tfidf_similarity(resume_text: str, job_text: str) -> float:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore[import-not-found]
        from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import-not-found]

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        vectors = vectorizer.fit_transform([resume_text, job_text])
        return float(cosine_similarity(vectors[0], vectors[1])[0][0])
    except Exception:
        resume_tokens = _token_set(resume_text)
        job_tokens = _token_set(job_text)
        if not resume_tokens or not job_tokens:
            return 0.0
        return len(resume_tokens & job_tokens) / len(resume_tokens | job_tokens)


def _token_set(text: str) -> set[str]:
    return {token for token in normalize_text(text).split() if token not in STOPWORDS and len(token) > 2}


def _heading_regex(headings: Iterable[str]) -> re.Pattern[str]:
    escaped = [re.escape(heading) for heading in headings]
    return re.compile(r"^\s*(?:" + "|".join(escaped) + r")\s*:?(?:\s*$|\s+)", re.I | re.M)


def _section_present(text: str, section: str) -> bool:
    headings = SECTION_HEADINGS[section]
    return bool(_heading_regex(headings).search(text))


def _evidence_for_section(text: str, section: str) -> int:
    normalized = normalize_text(text)
    if section == "education":
        terms = ["degree", "bachelor", "master", "university", "college", "gpa", "b.tech", "m.tech"]
    elif section == "experience":
        terms = ["experience", "intern", "engineer", "analyst", "manager", "company", "worked"]
    elif section == "projects":
        terms = ["project", "built", "developed", "implemented", "github", "portfolio"]
    else:
        terms = list(SKILL_ALIASES.keys())
    return sum(1 for term in terms if term in normalized)


def _quality_label(score: int) -> str:
    if score >= 3:
        return "Good"
    if score >= 1:
        return "Average"
    return "Poor"


def analyze_sections(resume_text: str) -> dict[str, dict[str, str]]:
    words = normalize_text(resume_text).split()
    action_count = len(set(words) & ACTION_VERBS)
    results = {}

    for section in ["education", "experience", "projects", "skills"]:
        score = 0
        has_heading = _section_present(resume_text, section)
        evidence = _evidence_for_section(resume_text, section)

        if has_heading:
            score += 2
        if evidence >= 2:
            score += 1
        if section in {"experience", "projects"} and action_count >= 3:
            score += 1

        label = _quality_label(score)
        results[section] = {
            "rating": label,
            "feedback": _section_feedback(section, label, has_heading, evidence, action_count),
        }

    return results


def _section_feedback(section: str, label: str, has_heading: bool, evidence: int, action_count: int) -> str:
    if label == "Good":
        if section in {"experience", "projects"}:
            return "The section is identifiable and includes role-relevant evidence with action-oriented wording."
        return "The section is identifiable and includes enough relevant details for screening."

    if not has_heading:
        return f"Add a clear '{section.title()}' heading so recruiters and ATS parsers can identify this content."

    if section in {"experience", "projects"} and action_count < 3:
        return "Use stronger action verbs and measurable outcomes to make the impact clearer."

    if evidence < 2:
        return "The section is present but needs more specific, role-relevant details."

    return "The section is usable but would benefit from clearer details and measurable outcomes."


def ats_score(resume_text: str, job_text: str, skill_match: SkillMatch) -> tuple[int, list[str]]:
    suggestions = []
    section_hits = sum(1 for section in SECTION_HEADINGS if _section_present(resume_text, section))
    heading_score = (section_hits / len(SECTION_HEADINGS)) * 35

    lines = [line.strip() for line in resume_text.splitlines() if line.strip()]
    avg_line_length = sum(len(line) for line in lines) / max(len(lines), 1)
    readability_score = 25 if avg_line_length <= 120 else 14

    bullet_count = sum(1 for line in lines if line.startswith(("-", "*", "•")))
    format_score = 20 if bullet_count >= 3 else 12 if bullet_count else 8

    keyword_score = min(20, (len(skill_match.matched) / max(len(skill_match.job_skills), 1)) * 20)
    score = round(heading_score + readability_score + format_score + keyword_score)

    if section_hits < len(SECTION_HEADINGS):
        suggestions.append("Use standard headings: Education, Experience, Projects, and Skills.")
    if skill_match.missing:
        suggestions.append("Add truthful evidence for missing job skills: " + ", ".join(skill_match.missing[:5]) + ".")
    if avg_line_length > 120:
        suggestions.append("Shorten dense lines and split responsibilities into readable bullets.")
    if bullet_count < 3:
        suggestions.append("Use bullet points for achievements, tools, and project outcomes.")

    return int(max(0, min(100, score))), suggestions[:3]


def bias_analysis(resume_text: str, job_text: str, original_similarity: float) -> dict[str, object]:
    anonymized_resume = anonymize_gender(resume_text)
    anonymized_job = anonymize_gender(job_text)
    anonymized_similarity = semantic_similarity(anonymized_resume, anonymized_job)
    score_delta = round(abs(original_similarity - anonymized_similarity), 2)

    resume_indicators = _detect_bias_terms(resume_text, GENDERED_TERMS | GENDERED_JOB_WORDS)
    job_indicators = _detect_bias_terms(job_text, GENDERED_TERMS | GENDERED_JOB_WORDS)
    indicator_count = len(resume_indicators) + len(job_indicators)

    if score_delta >= 8 or indicator_count >= 5:
        risk = "High"
    elif score_delta >= 3 or indicator_count:
        risk = "Medium"
    else:
        risk = "Low"

    suggestions = []
    if resume_indicators:
        suggestions.append("Remove gender-identifiable terms from the resume before scoring.")
    if job_indicators:
        suggestions.append("Rewrite gender-coded wording in the job description using role-specific requirements.")
    if score_delta >= 3:
        suggestions.append("Review anonymized and original scores because identity terms changed the similarity result.")
    if not suggestions:
        suggestions.append("Continue evaluating with anonymized text and job-relevant criteria only.")

    return {
        "risk_level": risk,
        "resume_indicators": resume_indicators,
        "job_indicators": job_indicators,
        "original_similarity": original_similarity,
        "anonymized_similarity": anonymized_similarity,
        "score_delta": score_delta,
        "suggestions": suggestions,
    }


def _detect_bias_terms(text: str, terms: set[str]) -> list[str]:
    normalized = normalize_text(text)
    found = []
    for term in sorted(terms):
        if _alias_pattern(term).search(normalized):
            found.append(term)
    return found


def final_score(
    semantic_score: float,
    skill_score: float,
    sections: dict[str, dict[str, str]],
    ats: int,
    resume_intelligence: dict[str, object] | None = None,
) -> int:
    intelligence = resume_intelligence if resume_intelligence is not None else {}
    section_score = float(
        intelligence.get(
            "weighted_section_score",
            sum(SECTION_SCORE_POINTS[item["rating"]] for item in sections.values()) / len(sections),
        )
    )
    impact_signal = min(100.0, float(intelligence.get("impact_count", 0)) * 25.0)
    tool_depth_signal = float(intelligence.get("average_tool_depth", 0.0))
    intelligence_score = (impact_signal * 0.55) + (tool_depth_signal * 0.45)
    weighted = (
        (semantic_score * 0.38)
        + (skill_score * 0.28)
        + (section_score * 0.18)
        + (ats * 0.13)
        + (intelligence_score * 0.03)
    )
    return int(round(max(0, min(100, weighted))))


def improvement_suggestions(
    skill_match: SkillMatch,
    sections: dict[str, dict[str, str]],
    job_text: str,
    resume_intelligence: dict[str, object] | None = None,
) -> list[str]:
    suggestions = []
    if skill_match.missing:
        suggestions.append("Add truthful experience or projects covering: " + ", ".join(skill_match.missing[:6]) + ".")
    if skill_match.partial:
        suggestions.append("Clarify partial matches by naming exact tools requested: " + ", ".join(skill_match.partial[:5]) + ".")
    graph_bridges = skill_graph_explanations(skill_match)
    if graph_bridges:
        suggestions.append("Related skill bridges found: " + "; ".join(graph_bridges[:3]) + ".")

    weak_sections = [name.title() for name, data in sections.items() if data["rating"] != "Good"]
    if weak_sections:
        suggestions.append("Improve these sections with clearer headings and concrete evidence: " + ", ".join(weak_sections) + ".")

    intelligence = resume_intelligence or {}
    if not intelligence.get("project_impacts"):
        suggestions.append("Add measurable project outcomes such as accuracy, latency, users, automation time saved, or revenue impact.")
    if int(intelligence.get("advanced_tool_count", 0)) == 0:
        suggestions.append("Show tool depth by describing what you built, deployed, optimized, or measured with the tools listed.")

    job_skills = extract_skills(job_text)
    if job_skills:
        suggestions.append("Mirror the role's terminology where accurate, especially: " + ", ".join(job_skills[:6]) + ".")

    suggestions.append("Use action verbs plus measurable outcomes, for example: 'Implemented X using Y, improving Z by N%.'")
    return suggestions[:5]


def analyze_resume(resume_text: str, job_text: str, use_trained_model: bool = True) -> dict[str, object]:
    semantic = semantic_similarity(resume_text, job_text)
    skill_match = compare_skills(resume_text, job_text)
    sections = analyze_sections(resume_text)
    ats, ats_suggestions = ats_score(resume_text, job_text, skill_match)
    resume_intelligence = analyze_resume_intelligence(resume_text, sections)
    formula_score = final_score(semantic, skill_match.score, sections, ats, resume_intelligence)
    trained_score = None
    scoring_method = "formula"
    if use_trained_model:
        features = extract_scoring_features(resume_text, job_text, semantic, skill_match, sections, ats)
        trained_score = predict_trained_match_score(features)
        if trained_score is not None:
            scoring_method = "trained_model"
    else:
        features = extract_scoring_features(resume_text, job_text, semantic, skill_match, sections, ats)
    score = trained_score if trained_score is not None else formula_score
    bias = bias_analysis(resume_text, job_text, semantic)
    improvements = improvement_suggestions(skill_match, sections, job_text, resume_intelligence)

    return {
        "final_score": score,
        "formula_score": formula_score,
        "trained_score": trained_score,
        "features": features,
        "scoring_method": scoring_method,
        "semantic_score": semantic,
        "skill_match": skill_match,
        "sections": sections,
        "resume_intelligence": resume_intelligence,
        "ats_score": ats,
        "ats_suggestions": ats_suggestions,
        "improvements": improvements,
        "bias": bias,
        "overall_explanation": _overall_explanation(score, semantic, skill_match, ats, sections, resume_intelligence),
    }


def extract_scoring_features(
    resume_text: str,
    job_text: str,
    semantic_score: float | None = None,
    skill_match: SkillMatch | None = None,
    sections: dict[str, dict[str, str]] | None = None,
    ats: int | None = None,
    use_transformer: bool = False,
) -> dict[str, float]:
    semantic = semantic_score if semantic_score is not None else semantic_similarity(resume_text, job_text, use_transformer)
    skills = skill_match if skill_match is not None else compare_skills(resume_text, job_text)
    section_data = sections if sections is not None else analyze_sections(resume_text)
    ats_value = ats if ats is not None else ats_score(resume_text, job_text, skills)[0]
    intelligence = analyze_resume_intelligence(resume_text, section_data)

    resume_tokens = _token_set(resume_text)
    job_tokens = _token_set(job_text)
    token_union = resume_tokens | job_tokens
    token_overlap = (len(resume_tokens & job_tokens) / len(token_union)) * 100 if token_union else 0.0
    resume_words = len(normalize_text(resume_text).split())
    job_words = len(normalize_text(job_text).split())
    length_ratio = min(resume_words, job_words) / max(resume_words, job_words, 1)

    features = {
        "semantic_score": float(semantic),
        "skill_score": float(skills.score),
        "matched_skill_count": float(len(skills.matched)),
        "missing_skill_count": float(len(skills.missing)),
        "partial_skill_count": float(len(skills.partial)),
        "resume_skill_count": float(len(skills.resume_skills)),
        "job_skill_count": float(len(skills.job_skills)),
        "ats_score": float(ats_value),
        "token_overlap": float(token_overlap),
        "resume_word_count": float(resume_words),
        "job_word_count": float(job_words),
        "length_ratio": float(length_ratio),
        "weighted_section_score": float(intelligence["weighted_section_score"]),
        "impact_count": float(intelligence["impact_count"]),
        "average_tool_depth": float(intelligence["average_tool_depth"]),
        "advanced_tool_count": float(intelligence["advanced_tool_count"]),
        "is_fresher": 1.0 if intelligence["candidate_stage"] == "fresher" else 0.0,
    }

    section_scores = {
        f"{section}_score": float(SECTION_SCORE_POINTS[data["rating"]])
        for section, data in section_data.items()
    }
    features.update(section_scores)
    features["section_average"] = sum(section_scores.values()) / max(len(section_scores), 1)
    return features


def feature_vector_from_dict(features: dict[str, float], feature_names: list[str]) -> list[float]:
    return [float(features.get(name, 0.0)) for name in feature_names]


@lru_cache(maxsize=1)
def load_trained_match_model(model_path: str = str(TRAINED_MODEL_PATH)) -> dict[str, object] | None:
    path = Path(model_path)
    if not path.exists():
        return None
    try:
        import joblib  # type: ignore[import-not-found]

        return joblib.load(path)
    except Exception:
        return None


def predict_trained_match_score(features: dict[str, float]) -> int | None:
    artifact = load_trained_match_model()
    if not artifact:
        return None

    model = artifact.get("model")
    feature_names = artifact.get("feature_names")
    if model is None or not isinstance(feature_names, list):
        return None

    prediction = model.predict([feature_vector_from_dict(features, feature_names)])[0]
    return int(round(max(0.0, min(100.0, float(prediction)))))


def learning_to_rank_scores(feature_dicts: list[dict[str, float]]) -> list[float] | None:
    artifact = load_trained_match_model()
    if not artifact or len(feature_dicts) < 2:
        return None

    ranker = artifact.get("ranker")
    feature_names = artifact.get("feature_names")
    if ranker is None or not isinstance(feature_names, list):
        return None
    if not hasattr(ranker, "predict_proba"):
        return None

    vectors = [feature_vector_from_dict(features, feature_names) for features in feature_dicts]
    wins = [0.0 for _ in vectors]
    comparisons = [0 for _ in vectors]

    for left_index in range(len(vectors)):
        for right_index in range(left_index + 1, len(vectors)):
            diff = [
                vectors[left_index][feature_index] - vectors[right_index][feature_index]
                for feature_index in range(len(feature_names))
            ]
            probability = float(ranker.predict_proba([diff])[0][1])
            wins[left_index] += probability
            wins[right_index] += 1.0 - probability
            comparisons[left_index] += 1
            comparisons[right_index] += 1

    return [
        round((wins[index] / comparisons[index]) * 100, 2) if comparisons[index] else 0.0
        for index in range(len(vectors))
    ]


def trained_model_info() -> dict[str, object]:
    artifact = load_trained_match_model()
    if not artifact:
        return {
            "available": False,
            "path": str(TRAINED_MODEL_PATH),
        }

    metrics = artifact.get("metrics", {})
    return {
        "available": True,
        "path": str(TRAINED_MODEL_PATH),
        "trained_rows": artifact.get("trained_rows"),
        "target_scale": artifact.get("target_scale"),
        "model_type": artifact.get("model_type", "regression"),
        "metrics": metrics,
        "rank_metrics": artifact.get("rank_metrics", {"available": False}),
    }


def rank_jobs_for_resume(
    resume_text: str,
    jobs: Iterable[JobPosting | dict[str, str]],
    limit: int | None = None,
) -> list[dict[str, object]]:
    ranked_jobs = []

    for index, job in enumerate(jobs, start=1):
        posting = _coerce_job_posting(job)
        if not posting.description.strip():
            continue

        result = analyze_resume(resume_text, posting.description)
        experience = experience_fit(resume_text, posting.description)
        if experience["penalty"]:
            result = dict(result)
            original_score = int(result["final_score"])
            result["final_score"] = max(0, original_score - int(experience["penalty"]))
            result["experience_adjusted_from"] = original_score
        result["experience_fit"] = experience
        skill_match: SkillMatch = result["skill_match"]  # type: ignore[assignment]
        ranked_jobs.append(
            {
                "rank": 0,
                "source_index": index,
                "title": posting.title or f"Job {index}",
                "company": posting.company,
                "url": posting.url,
                "description": posting.description,
                "final_score": result["final_score"],
                "semantic_score": result["semantic_score"],
                "skill_score": skill_match.score,
                "matched_skills": skill_match.matched,
                "missing_skills": skill_match.missing,
                "partial_skills": skill_match.partial,
                "ats_score": result["ats_score"],
                "bias_risk": result["bias"]["risk_level"],  # type: ignore[index]
                "experience_fit": experience,
                "summary": "",
                "analysis": result,
                "_posting": posting,
            }
        )

    ltr_scores = learning_to_rank_scores(
        [
            item["analysis"]["features"]  # type: ignore[index]
            for item in ranked_jobs
            if isinstance(item.get("analysis"), dict)
        ]
    )
    if ltr_scores and len(ltr_scores) == len(ranked_jobs):
        for item, ltr_score in zip(ranked_jobs, ltr_scores):
            base_score = int(item["final_score"])
            blended_score = int(round((0.65 * ltr_score) + (0.35 * base_score)))
            item["base_final_score"] = base_score
            item["learning_to_rank_score"] = ltr_score
            item["ranking_score"] = blended_score
            item["final_score"] = max(0, min(100, blended_score))
            item["analysis"]["final_score"] = item["final_score"]  # type: ignore[index]
            item["analysis"]["learning_to_rank_score"] = ltr_score  # type: ignore[index]
            item["analysis"]["base_final_score"] = base_score  # type: ignore[index]
            item["analysis"]["scoring_method"] = "learning_to_rank"  # type: ignore[index]
    else:
        for item in ranked_jobs:
            item["ranking_score"] = item["final_score"]

    ranked_jobs.sort(key=lambda item: (int(item["ranking_score"]), float(item["semantic_score"])), reverse=True)
    for rank, item in enumerate(ranked_jobs, start=1):
        item["rank"] = rank
        posting = item.pop("_posting")
        item["summary"] = explain_job_match(item["analysis"], posting)  # type: ignore[arg-type]

    return ranked_jobs[:limit] if limit is not None else ranked_jobs


def _coerce_job_posting(job: JobPosting | dict[str, str]) -> JobPosting:
    if isinstance(job, JobPosting):
        return job

    return JobPosting(
        description=job.get("description", "") or job.get("job_description", ""),
        title=job.get("title", "") or job.get("job_title", "") or job.get("role", ""),
        company=job.get("company", "") or job.get("organization", ""),
        url=job.get("url", "") or job.get("link", ""),
    )


def explain_job_match(result: dict[str, object], posting: JobPosting) -> str:
    skill_match: SkillMatch = result["skill_match"]  # type: ignore[assignment]
    matched = ", ".join(skill_match.matched[:5]) if skill_match.matched else "no extracted required skills"
    missing = ", ".join(skill_match.missing[:4]) if skill_match.missing else "no major extracted skill gaps"
    graph_bridges = skill_graph_explanations(skill_match)
    graph_note = f" Related skills also bridge {graph_bridges[0]}." if graph_bridges else ""
    title = posting.title or "This job"
    method_labels = {
        "learning_to_rank": "learning-to-rank model",
        "trained_model": "trained model",
    }
    scoring_method = method_labels.get(str(result.get("scoring_method")), "formula")
    experience = result.get("experience_fit")
    experience_note = ""
    if isinstance(experience, dict) and experience.get("warning"):
        experience_note = f" {experience['warning']}"
    return (
        f"{title} ranks here because the resume matches {matched}, has {missing}, "
        f"and scored {result['semantic_score']}% semantic alignment with {result['ats_score']}/100 ATS compatibility. "
        f"The final score uses the {scoring_method}.{graph_note}{experience_note}"
    )


def _overall_explanation(
    score: int,
    semantic: float,
    skill_match: SkillMatch,
    ats: int,
    sections: dict[str, dict[str, str]],
    resume_intelligence: dict[str, object] | None = None,
) -> str:
    weak_sections = [name.title() for name, data in sections.items() if data["rating"] == "Poor"]
    missing = ", ".join(skill_match.missing[:4]) if skill_match.missing else "no major extracted job skills"
    intelligence = resume_intelligence or {}
    stage = str(intelligence.get("candidate_stage", "professional"))
    section_score = float(intelligence.get("weighted_section_score", 0.0))
    impact_count = int(intelligence.get("impact_count", 0))
    graph_bridges = skill_graph_explanations(skill_match)
    graph_sentence = (
        " Related skills partially cover: " + "; ".join(graph_bridges[:2]) + "."
        if graph_bridges
        else ""
    )
    return (
        f"The resume scores {score}/100 because semantic alignment is {semantic:.0f}%, "
        f"skill coverage is {skill_match.score:.0f}%, and ATS compatibility is {ats}/100. "
        f"Section quality is weighted for a {stage} profile ({section_score:.0f}/100), with {impact_count} measurable impact signals. "
        f"The main gap is {missing}; "
        f"{'weak sections include ' + ', '.join(weak_sections) if weak_sections else 'the core resume sections are mostly identifiable'}."
        f"{graph_sentence}"
    )


def format_report(result: dict[str, object]) -> str:
    skill_match: SkillMatch = result["skill_match"]  # type: ignore[assignment]
    sections: dict[str, dict[str, str]] = result["sections"]  # type: ignore[assignment]
    bias: dict[str, object] = result["bias"]  # type: ignore[assignment]
    resume_intelligence: dict[str, object] = result["resume_intelligence"]  # type: ignore[assignment]
    tool_depths = resume_intelligence.get("tool_depths", {})
    tool_depth_items = [
        f"{skill}: {depth}"
        for skill, depth in list(tool_depths.items())[:8]
    ] if isinstance(tool_depths, dict) else []
    method_labels = {
        "learning_to_rank": "Learning-to-rank model",
        "trained_model": "Trained model",
    }
    scoring_method = method_labels.get(str(result.get("scoring_method")), "Formula fallback")

    return f"""📊 **Final Resume Score:** {result["final_score"]}/100

Scoring Method: {scoring_method}

 **Semantic Match Score:** {result["semantic_score"]}%

---

 **Matched Skills:**

{_bullet_list(skill_match.matched)}

 **Missing Skills:**

{_bullet_list(skill_match.missing)}

 **Partial Matches:**

{_bullet_list(skill_match.partial)}

---

 **Section-wise Analysis:**

**Education:** {sections["education"]["rating"]}
Feedback: {sections["education"]["feedback"]}

**Experience:** {sections["experience"]["rating"]}
Feedback: {sections["experience"]["feedback"]}

**Projects:** {sections["projects"]["rating"]}
Feedback: {sections["projects"]["feedback"]}

**Skills:** {sections["skills"]["rating"]}
Feedback: {sections["skills"]["feedback"]}

---

 **Resume Intelligence:**
Candidate Stage: {resume_intelligence["candidate_stage"]}
Weighted Section Score: {resume_intelligence["weighted_section_score"]}/100
Measurable Project Impacts:
{_bullet_list(resume_intelligence["project_impacts"])}
Tool Depth:
{_bullet_list(tool_depth_items)}

---

 **ATS Compatibility Score:** {result["ats_score"]}/100
Suggestions:

{_bullet_list(result["ats_suggestions"])}

---

 **Improvement Suggestions:**

{_bullet_list(result["improvements"])}

---

 **Bias Risk Level:** {bias["risk_level"]}
 **Detected Bias Indicators:** {_bias_indicator_text(bias)}
💡**Suggestions to Reduce Bias:** {"; ".join(bias["suggestions"])}
Score comparison: original semantic score {bias["original_similarity"]}%, anonymized semantic score {bias["anonymized_similarity"]}%.

---

 **Overall Explanation:**
{result["overall_explanation"]}
"""


def _bullet_list(items: object) -> str:
    values = list(items) if isinstance(items, Iterable) and not isinstance(items, str) else []
    if not values:
        return "* None detected from the provided text."
    return "\n".join(f"* {item}" for item in values)


def _bias_indicator_text(bias: dict[str, object]) -> str:
    resume_terms = bias.get("resume_indicators", [])
    job_terms = bias.get("job_indicators", [])
    parts = []
    if resume_terms:
        parts.append("resume: " + ", ".join(resume_terms))
    if job_terms:
        parts.append("job description: " + ", ".join(job_terms))
    if not parts:
        return "None detected."
    return "; ".join(parts)


def analyze_dataset(csv_path: Path, limit: int | None = None) -> list[dict[str, object]]:
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if limit is not None and index >= limit:
                break
            result = analyze_resume(row.get("resume", ""), row.get("job_description", ""))
            result["dataset_match_score"] = row.get("match_score")
            rows.append(result)
    return rows


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a resume against a job description.")
    parser.add_argument("--resume-file", type=Path, help="Path to a plain-text resume file.")
    parser.add_argument("--job-file", type=Path, help="Path to a plain-text job description file.")
    parser.add_argument("--resume-text", help="Resume text.")
    parser.add_argument("--job-text", help="Job description text.")
    args = parser.parse_args()

    resume_text = _read_text(args.resume_file) if args.resume_file else args.resume_text
    job_text = _read_text(args.job_file) if args.job_file else args.job_text

    if not resume_text or not job_text:
        raise SystemExit("Provide both resume and job description using --resume-file/--job-file or --resume-text/--job-text.")

    print(format_report(analyze_resume(resume_text, job_text)))


if __name__ == "__main__":
    main()
