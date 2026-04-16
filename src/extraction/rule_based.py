"""
Rule-based skill extraction using spaCy PhraseMatcher.
Vocabulary is built from O*NET Tools & Technology + a curated tech skills list.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import spacy
from spacy.matcher import PhraseMatcher

# ---------------------------------------------------------------------------
# Curated software / tech skill vocabulary
# ---------------------------------------------------------------------------
CORE_TECH_SKILLS: list[str] = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go",
    "golang", "rust", "swift", "kotlin", "php", "r", "scala", "matlab",
    "perl", "haskell", "elixir", "clojure", "dart", "lua", "groovy",
    # Web frontend
    "react", "angular", "vue", "vue.js", "next.js", "nuxt", "svelte",
    "html", "css", "sass", "scss", "tailwind", "bootstrap", "webpack",
    "vite", "redux", "graphql",
    # Web backend / frameworks
    "node.js", "django", "flask", "fastapi", "spring", "spring boot",
    "express", "rails", "laravel", "asp.net", ".net", "nestjs",
    # Databases
    "sql", "mysql", "postgresql", "postgres", "mongodb", "redis",
    "elasticsearch", "cassandra", "oracle", "sqlite", "dynamodb",
    "neo4j", "cockroachdb", "mariadb", "firebase",
    # Cloud & DevOps
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "github actions", "gitlab ci",
    "ci/cd", "helm", "prometheus", "grafana", "nginx", "apache",
    # ML / AI / Data
    "machine learning", "deep learning", "neural networks", "nlp",
    "computer vision", "reinforcement learning",
    "tensorflow", "pytorch", "scikit-learn", "keras", "hugging face",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "spark", "hadoop", "kafka", "airflow", "dbt", "mlflow",
    "langchain", "openai", "llm",
    # Data warehousing / BI
    "tableau", "power bi", "looker", "snowflake", "bigquery", "redshift",
    "databricks",
    # Mobile
    "react native", "flutter", "ios", "android", "xcode",
    # Security
    "penetration testing", "ethical hacking", "owasp", "siem", "oauth",
    "jwt", "ssl", "tls", "encryption",
    # Methodologies & tools
    "agile", "scrum", "kanban", "jira", "confluence", "trello",
    "rest api", "restful", "microservices", "event-driven", "grpc",
    "git", "github", "gitlab", "bitbucket", "svn",
    "linux", "unix", "bash", "shell scripting", "powershell",
    "object-oriented programming", "oop", "design patterns",
    "test-driven development", "tdd", "bdd",
    "unit testing", "pytest", "junit", "selenium", "cypress",
    # General data skills
    "data analysis", "data engineering", "data science",
    "etl", "data pipeline", "feature engineering", "a/b testing",
    "statistics", "probability", "linear algebra",
    # Soft / process
    "api design", "system design", "code review", "technical documentation",
]

# Regex patterns for multi-word tech terms not easily caught by PhraseMatcher
_REGEX_PATTERNS: list[re.Pattern] = [
    re.compile(r"\bci[/\-]cd\b", re.I),
    re.compile(r"\brest(?:ful)?\s*api\b", re.I),
    re.compile(r"\bnode\.js\b", re.I),
    re.compile(r"\bvue\.js\b", re.I),
    re.compile(r"\bnext\.js\b", re.I),
    re.compile(r"\basp\.net\b", re.I),
    re.compile(r"\b\.net\s*(core|framework)?\b", re.I),
    re.compile(r"\bc\+\+\b"),
    re.compile(r"\bc#\b"),
]


# ---------------------------------------------------------------------------
# Lazy-loaded spaCy model and matcher
# ---------------------------------------------------------------------------
_nlp: spacy.Language | None = None
_matcher: PhraseMatcher | None = None


def _get_nlp() -> spacy.Language:
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm", disable=["ner", "parser", "lemmatizer"])
    return _nlp


def _get_matcher(extra_skills: list[str] | None = None) -> PhraseMatcher:
    global _matcher
    if _matcher is not None:
        return _matcher

    nlp = _get_nlp()
    _matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    vocab = list(set(CORE_TECH_SKILLS + (extra_skills or [])))
    patterns = list(nlp.pipe(vocab))
    _matcher.add("SKILL", patterns)
    return _matcher


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_vocab_from_onet(onet_skills: list[str]) -> list[str]:
    """Filter O*NET tool examples to likely software skills."""
    sw_terms = re.compile(
        r"software|python|java|sql|linux|cloud|docker|aws|azure|data|api|web|"
        r"javascript|typescript|react|node|django|kubernetes|spark|hadoop",
        re.I,
    )
    return [s for s in onet_skills if sw_terms.search(s)]


def extract_skills(
    text: str,
    extra_skills: list[str] | None = None,
    max_chars: int = 100_000,
) -> list[dict]:
    """
    Extract skills from text using PhraseMatcher + regex fallback.

    Returns a list of dicts:
        {skill, canonical, source_span, method}
    """
    if not text or not isinstance(text, str):
        return []

    text = text[:max_chars]
    nlp = _get_nlp()
    matcher = _get_matcher(extra_skills)
    doc = nlp(text)

    seen: set[str] = set()
    skills: list[dict] = []

    # PhraseMatcher hits
    for _, start, end in matcher(doc):
        span = doc[start:end]
        canonical = span.text.lower().strip()
        if canonical and canonical not in seen:
            seen.add(canonical)
            skills.append(
                {
                    "skill": span.text,
                    "canonical": canonical,
                    "source_span": span.text,
                    "method": "rule_based",
                }
            )

    # Regex fallback for special tokens (c++, c#, .net, etc.)
    for pattern in _REGEX_PATTERNS:
        for m in pattern.finditer(text):
            canonical = m.group().lower().strip()
            if canonical and canonical not in seen:
                seen.add(canonical)
                skills.append(
                    {
                        "skill": m.group(),
                        "canonical": canonical,
                        "source_span": m.group(),
                        "method": "regex",
                    }
                )

    return skills


def extract_skills_from_section(text: str, extra_skills: list[str] | None = None) -> list[dict]:
    """
    Extract skills preferring the skills section of a resume if detected,
    then supplement with the full text.
    """
    from src.data.preprocess import extract_skills_section

    skills_text = extract_skills_section(text)
    found = extract_skills(skills_text, extra_skills)
    found_canonical = {s["canonical"] for s in found}

    # Supplement with full-text extraction
    for s in extract_skills(text, extra_skills):
        if s["canonical"] not in found_canonical:
            found.append(s)
            found_canonical.add(s["canonical"])

    return found
