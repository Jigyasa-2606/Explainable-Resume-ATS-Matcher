import re
from typing import Dict, List


SECTION_PATTERNS = {
    "skills": [r"\bskills?\b", r"\btechnical skills?\b", r"\bcore competencies\b"],
    "experience": [r"\bexperience\b", r"\bwork experience\b", r"\bemployment\b"],
    "education": [r"\beducation\b", r"\bacademic\b", r"\bqualifications?\b"],
    "projects": [r"\bprojects?\b", r"\bpersonal projects?\b", r"\bkey projects?\b"],
}


def _line_matches_header(line: str, patterns: List[str]) -> bool:
    stripped = line.strip().lower()
    if not stripped:
        return False
    # Limit false positives by checking short header-like lines.
    if len(stripped.split()) > 6:
        return False
    return any(re.search(pattern, stripped) for pattern in patterns)


def segment_resume_sections(raw_text: str) -> Dict[str, str]:
    lines = [line.strip() for line in (raw_text or "").splitlines()]
    sections = {key: [] for key in SECTION_PATTERNS.keys()}

    active_section = None
    for line in lines:
        matched_section = None
        for section_name, patterns in SECTION_PATTERNS.items():
            if _line_matches_header(line, patterns):
                matched_section = section_name
                break

        if matched_section:
            active_section = matched_section
            continue

        if active_section and line:
            sections[active_section].append(line)

    result = {name: "\n".join(content).strip() for name, content in sections.items()}
    result["full_text"] = raw_text or ""
    return result
