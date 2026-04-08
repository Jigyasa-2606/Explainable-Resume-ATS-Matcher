import re
from typing import Dict, List, Set

from src.features.build_phase2_features import SKILL_PHRASES, _skills_in_text


def _infer_role_skills(text: str) -> Set[str]:
    return _skills_in_text(text.lower(), SKILL_PHRASES)


def generate_actionable_suggestions(
    job_description_clean: str,
    resume_clean_text: str,
    negative_features: List[Dict[str, float]],
) -> List[str]:
    suggestions: List[str] = []
    jd_skills = _infer_role_skills(job_description_clean)
    resume_skills = _infer_role_skills(resume_clean_text)
    missing = sorted(jd_skills - resume_skills)

    if missing:
        missing_preview = ", ".join(missing[:6])
        suggestions.append(
            f"Add evidence for missing JD skills in experience/projects: {missing_preview}."
        )

    for item in negative_features:
        feat = item.get("feature", "")
        if feat == "skill_jd_coverage_ratio":
            suggestions.append("Increase alignment with JD skills by adding direct skill mentions and project proof.")
        elif feat == "keyword_overlap_ratio":
            suggestions.append("Mirror key JD terms in your resume bullet points where genuinely applicable.")
        elif feat == "experience_year_gap_abs":
            suggestions.append("Highlight relevant years of experience explicitly (e.g., '3+ years').")
        elif feat == "jd_resume_cosine_clean":
            suggestions.append("Rewrite summary and experience bullets to match JD language and responsibilities.")
        elif re.search(r"\b(?:aws|gcp|azure|docker|kubernetes|mlops|python|sql)\b", feat):
            suggestions.append(f"Add practical evidence for '{feat}' in projects or work experience.")

    # Deduplicate while preserving order.
    deduped = list(dict.fromkeys(suggestions))
    return deduped[:8]
