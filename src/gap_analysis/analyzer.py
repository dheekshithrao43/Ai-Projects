"""
Skill-gap analysis module.
Compares candidate's extracted skills against job-required skills
and classifies each gap as matched / partial / missing.
"""

from __future__ import annotations

from rapidfuzz import fuzz


def analyze_gap(
    candidate_skills: list[str],
    job_skills: list[str],
    match_threshold: int = 85,
    partial_threshold: int = 65,
) -> dict:
    """
    Compare candidate skills against required job skills.

    Args:
        candidate_skills: Canonical skill strings from the CV.
        job_skills: Canonical skill strings from the job description.
        match_threshold: Fuzzy score (0-100) to count as a full match.
        partial_threshold: Fuzzy score to count as a partial match.

    Returns:
        {
            matched:  [{"job_skill", "candidate_match", "score"}],
            partial:  [{"job_skill", "candidate_match", "score"}],
            missing:  [str],
            fit_score:    float in [0, 1]  (matched + 0.5*partial) / total,
            coverage_pct: float in [0, 100],
            total_job_skills: int,
        }
    """
    candidate_lower = [s.lower().strip() for s in candidate_skills if s]
    job_lower = [s.lower().strip() for s in job_skills if s]

    if not job_lower:
        return {
            "matched": [],
            "partial": [],
            "missing": [],
            "fit_score": 0.0,
            "coverage_pct": 0.0,
            "total_job_skills": 0,
        }

    matched: list[dict] = []
    partial: list[dict] = []
    missing: list[str] = []

    for job_skill in job_lower:
        best_score = 0
        best_match = ""

        for cand_skill in candidate_lower:
            score = fuzz.token_sort_ratio(job_skill, cand_skill)
            if score > best_score:
                best_score = score
                best_match = cand_skill

        if best_score >= match_threshold:
            matched.append(
                {"job_skill": job_skill, "candidate_match": best_match, "score": best_score}
            )
        elif best_score >= partial_threshold:
            partial.append(
                {"job_skill": job_skill, "candidate_match": best_match, "score": best_score}
            )
        else:
            missing.append(job_skill)

    total = len(job_lower)
    fit_score = (len(matched) + 0.5 * len(partial)) / total if total else 0.0
    coverage_pct = (len(matched) + len(partial)) / total * 100 if total else 0.0

    return {
        "matched": matched,
        "partial": partial,
        "missing": missing,
        "fit_score": round(fit_score, 3),
        "coverage_pct": round(coverage_pct, 1),
        "total_job_skills": total,
    }


def prioritise_upskilling(missing: list[str], partial: list[dict]) -> list[dict]:
    """
    Produce a prioritised upskilling list from missing and partial skills.

    Missing skills (score 0) get priority 1 (highest).
    Partial skills are ranked by how far below the match threshold they are.
    """
    recommendations: list[dict] = []

    for skill in missing:
        recommendations.append(
            {"skill": skill, "status": "missing", "priority": 1, "current_match": None}
        )

    for item in sorted(partial, key=lambda x: x["score"]):
        recommendations.append(
            {
                "skill": item["job_skill"],
                "status": "partial",
                "priority": 2,
                "current_match": item["candidate_match"],
            }
        )

    return recommendations


def format_gap_summary(gap: dict) -> str:
    """Return a human-readable one-line summary of the gap analysis."""
    total = gap["total_job_skills"]
    n_matched = len(gap["matched"])
    n_partial = len(gap["partial"])
    n_missing = len(gap["missing"])
    return (
        f"{n_matched}/{total} skills matched | "
        f"{n_partial} partial | "
        f"{n_missing} missing | "
        f"Fit score: {gap['fit_score']:.0%}"
    )
