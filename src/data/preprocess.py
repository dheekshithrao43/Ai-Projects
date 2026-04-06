"""Text cleaning and CV section detection."""

import re
import unicodedata

SECTION_HEADERS = {
    "skills": re.compile(
        r"^\s*(technical\s+)?skills?\s*:?\s*$", re.I | re.M
    ),
    "experience": re.compile(
        r"^\s*(work\s+)?(experience|employment|history)\s*:?\s*$", re.I | re.M
    ),
    "education": re.compile(
        r"^\s*(education|qualifications?|academic\s+background)\s*:?\s*$", re.I | re.M
    ),
    "summary": re.compile(
        r"^\s*(professional\s+)?(summary|objective|profile|about)\s*:?\s*$", re.I | re.M
    ),
    "certifications": re.compile(
        r"^\s*(certifications?|courses?|training|licenses?)\s*:?\s*$", re.I | re.M
    ),
    "projects": re.compile(
        r"^\s*projects?\s*:?\s*$", re.I | re.M
    ),
}


def clean_text(text: str) -> str:
    """Normalize whitespace and strip non-ASCII noise."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def detect_sections(text: str) -> dict[str, str]:
    """
    Split resume text into named sections.
    Returns a dict mapping section name -> section text.
    """
    lines = text.split("\n")
    sections: dict[str, list[str]] = {"other": []}
    current = "other"

    for line in lines:
        matched_section = None
        for section, pattern in SECTION_HEADERS.items():
            if pattern.match(line):
                matched_section = section
                break

        if matched_section:
            sections.setdefault(matched_section, [])
            current = matched_section
        else:
            sections.setdefault(current, []).append(line)

    return {k: "\n".join(v).strip() for k, v in sections.items() if v}


def extract_skills_section(text: str) -> str:
    """Return the skills section text, or the full text if no section found."""
    sections = detect_sections(text)
    return sections.get("skills", text)


def parse_skills_string(skills_str: str) -> list[str]:
    """
    Parse a comma/semicolon/pipe/newline-delimited skills string into a clean list.
    Also handles parenthetical examples like 'Frameworks (e.g., React, Angular)'.
    """
    if not isinstance(skills_str, str) or not skills_str.strip():
        return []
    # Expand parenthetical examples: keep content inside parens
    skills_str = re.sub(r"\(e\.g\.?,?\s*", "(", skills_str)
    skills_str = re.sub(r"[()]", ",", skills_str)
    skills = re.split(r"[,;|\n]+", skills_str)
    cleaned = []
    for s in skills:
        s = s.strip().lower()
        # Drop long phrases (likely descriptions, not skill names)
        if s and len(s.split()) <= 5 and len(s) > 1:
            cleaned.append(s)
    return cleaned
