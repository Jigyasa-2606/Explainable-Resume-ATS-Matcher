from typing import Dict, List

from src.features.build_phase2_features import SKILL_PHRASES, _skills_in_text
from src.preprocess.text_cleaning import clean_text


def analyze_skill_gap(job_description: str, resume_text: str) -> Dict[str, object]:
    jd_clean = clean_text(job_description)
    resume_clean = clean_text(resume_text)

    jd_skills = _skills_in_text(jd_clean, SKILL_PHRASES)
    resume_skills = _skills_in_text(resume_clean, SKILL_PHRASES)

    matched = sorted(jd_skills.intersection(resume_skills))
    missing = sorted(jd_skills - resume_skills)
    extra = sorted(resume_skills - jd_skills)
    coverage = (len(matched) / max(len(jd_skills), 1)) * 100.0

    return {
        "matched_skills": matched,
        "missing_skills": missing,
        "extra_skills": extra,
        "jd_skills": sorted(jd_skills),
        "resume_skills": sorted(resume_skills),
        "coverage_percent": round(coverage, 2),
    }


def top_missing_skill_suggestions(missing_skills: List[str]) -> List[str]:
    suggestions = []
    for skill in missing_skills[:8]:
        suggestions.append(
            f"Add a project or work bullet showing measurable impact with {skill}."
        )
    return suggestions
