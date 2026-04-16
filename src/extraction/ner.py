"""
NER-based extraction for job titles, organisations, and degrees.
Uses spaCy's en_core_web_sm for entity recognition,
supplemented with regex patterns for software job titles.
"""

from __future__ import annotations

import re

import spacy

_nlp: spacy.Language | None = None

JOB_TITLE_PATTERN = re.compile(
    r"\b(?:senior|junior|lead|principal|staff|associate|mid[\-\s]?level)?\s*"
    r"(?:software|backend|frontend|full[\-\s]?stack|data|machine\s+learning|ml|ai|"
    r"devops|cloud|security|mobile|ios|android|platform|site\s+reliability|sre|"
    r"embedded|systems?)\s*"
    r"(?:engineer|developer|architect|scientist|analyst|specialist|"
    r"manager|lead|consultant|intern)\b",
    re.I,
)

DEGREE_PATTERN = re.compile(
    r"\b(?:b\.?sc|b\.?eng|b\.?tech|bachelor(?:'s)?|master(?:'s)?|m\.?sc|"
    r"m\.?eng|phd|ph\.?d|doctorate|mba|associate(?:'s)?)\b",
    re.I,
)


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def extract_entities(text: str, max_chars: int = 50_000) -> dict:
    """
    Extract structured entities from resume or job description text.

    Returns:
        {
            orgs: list of organisation names,
            titles: list of job titles found,
            degrees: list of degree mentions,
        }
    """
    if not text or not isinstance(text, str):
        return {"orgs": [], "titles": [], "degrees": []}

    text = text[:max_chars]
    nlp = _get_nlp()
    doc = nlp(text)

    orgs: set[str] = set()
    for ent in doc.ents:
        if ent.label_ == "ORG" and len(ent.text) > 2:
            orgs.add(ent.text.strip())

    titles: set[str] = set()
    for m in JOB_TITLE_PATTERN.finditer(text):
        titles.add(m.group().strip())

    degrees: set[str] = set()
    for m in DEGREE_PATTERN.finditer(text):
        degrees.add(m.group().strip())

    return {
        "orgs": sorted(orgs),
        "titles": sorted(titles),
        "degrees": sorted(degrees),
    }


def extract_years_experience(text: str) -> int | None:
    """
    Attempt to parse years of experience from free text.
    e.g. "5+ years of experience", "3-5 years experience"
    Returns the maximum found value, or None.
    """
    pattern = re.compile(
        r"(\d+)\+?\s*(?:to|\-|–)?\s*(\d+)?\s*years?\s*(?:of\s+)?experience",
        re.I,
    )
    values: list[int] = []
    for m in pattern.finditer(text):
        values.append(int(m.group(1)))
        if m.group(2):
            values.append(int(m.group(2)))

    return max(values) if values else None
