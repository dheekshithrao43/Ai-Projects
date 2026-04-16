"""
Skill normalisation: maps raw extracted skills to canonical labels
using the O*NET Tools & Technology vocabulary and fuzzy matching.
"""

from __future__ import annotations

from rapidfuzz import fuzz, process

from src.extraction.rule_based import CORE_TECH_SKILLS

# Common abbreviations and aliases → canonical form
SKILL_ALIASES: dict[str, str] = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "aws": "aws",
    "amazon web services": "aws",
    "gcp": "gcp",
    "google cloud platform": "gcp",
    "google cloud": "gcp",
    "azure": "azure",
    "microsoft azure": "azure",
    "k8s": "kubernetes",
    "js": "javascript",
    "ts": "typescript",
    "pg": "postgresql",
    "postgres": "postgresql",
    "node": "node.js",
    "nodejs": "node.js",
    "vue": "vue",
    "vuejs": "vue",
    "reactjs": "react",
    "react.js": "react",
    "c++": "c++",
    "cpp": "c++",
    "csharp": "c#",
    "dotnet": ".net",
    "dot net": ".net",
    "oop": "object-oriented programming",
    "tdd": "test-driven development",
    "ci/cd": "ci/cd",
    "cicd": "ci/cd",
    "rest": "rest api",
    "restful": "rest api",
    "restful api": "rest api",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "hf": "hugging face",
    "llm": "llm",
    "large language model": "llm",
    "sql server": "sql",
    "mssql": "sql",
    "nosql": "mongodb",
    "elastic": "elasticsearch",
    "elk": "elasticsearch",
    "shell": "bash",
    "shell scripting": "bash",
    "linux/unix": "linux",
    "unix/linux": "linux",
}

_canonical_vocab: list[str] = []


def _load_canonical_vocab() -> list[str]:
    """
    Build canonical skill vocabulary from O*NET tech skills + core list.
    Cached after first call.
    """
    global _canonical_vocab
    if _canonical_vocab:
        return _canonical_vocab

    vocab: set[str] = set(CORE_TECH_SKILLS)

    try:
        from src.data.loader import load_onet_tech_skills
        from src.extraction.rule_based import build_vocab_from_onet

        onet_skills = load_onet_tech_skills()
        sw_onet = build_vocab_from_onet(onet_skills)
        vocab.update(sw_onet)
    except Exception:
        pass  # fall back to core list only

    _canonical_vocab = sorted(vocab)
    return _canonical_vocab


def normalise_skill(raw: str, threshold: int = 75) -> dict:
    # Alias lookup first (fast, exact)
    alias = SKILL_ALIASES.get(raw.lower().strip())
    if alias:
        return {"raw": raw, "canonical": alias, "confidence": 1.0}

    """
    Normalise a single raw skill string to its canonical form.

    Returns:
        {raw, canonical, confidence}
    """
    canonical_list = _load_canonical_vocab()
    raw_lower = raw.lower().strip()

    # Exact match first
    if raw_lower in canonical_list:
        return {"raw": raw, "canonical": raw_lower, "confidence": 1.0}

    result = process.extractOne(
        raw_lower,
        canonical_list,
        scorer=fuzz.token_sort_ratio,
    )

    if result and result[1] >= threshold:
        canonical_label, score, _ = result
        return {"raw": raw, "canonical": canonical_label, "confidence": round(score / 100, 3)}

    # Below threshold — keep raw as canonical with low confidence
    return {"raw": raw, "canonical": raw_lower, "confidence": 0.4}


def normalise_skills(
    skills: list[str | dict],
    threshold: int = 75,
) -> list[dict]:
    """
    Normalise a list of raw skills.

    Accepts either:
        - list of strings: ["python", "AWS"]
        - list of dicts from rule_based.extract_skills: [{"skill": ..., "canonical": ...}]

    Returns list of {raw, canonical, confidence}.
    """
    normalised: list[dict] = []
    seen: set[str] = set()

    for item in skills:
        if isinstance(item, dict):
            raw = item.get("canonical") or item.get("skill", "")
        else:
            raw = item

        if not raw or not isinstance(raw, str):
            continue

        result = normalise_skill(raw, threshold)
        canonical = result["canonical"]

        if canonical not in seen:
            seen.add(canonical)
            normalised.append(result)

    return normalised


def deduplicate_skills(skills: list[dict], similarity_threshold: int = 90) -> list[dict]:
    """
    Remove near-duplicate skills from a normalised skill list.
    Keeps the entry with higher confidence when duplicates are found.
    """
    kept: list[dict] = []
    for skill in skills:
        is_dup = False
        for existing in kept:
            sim = fuzz.token_sort_ratio(skill["canonical"], existing["canonical"])
            if sim >= similarity_threshold:
                # Keep whichever has higher confidence
                if skill["confidence"] > existing["confidence"]:
                    kept.remove(existing)
                    kept.append(skill)
                is_dup = True
                break
        if not is_dup:
            kept.append(skill)
    return kept
