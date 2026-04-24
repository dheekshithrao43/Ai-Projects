"""
Evaluation metrics for the matching and skill extraction pipelines.

Matching metrics: Precision@K, Recall@K, MRR, NDCG@K
Classification metrics: Precision, Recall, F1 (macro + weighted)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

LABEL_MAP = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
LABEL_NAMES = ["No Fit", "Potential Fit", "Good Fit"]


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def precision_at_k(relevant: set, retrieved: list, k: int) -> float:
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / k if k else 0.0


def recall_at_k(relevant: set, retrieved: list, k: int) -> float:
    if not relevant:
        return 0.0
    hits = sum(1 for r in retrieved[:k] if r in relevant)
    return hits / len(relevant)


def mrr(relevant: set, retrieved: list) -> float:
    """Mean Reciprocal Rank for a single query."""
    for rank, item in enumerate(retrieved, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(relevance_scores: list[float], k: int) -> float:
    scores = relevance_scores[:k]
    return sum(rel / np.log2(i + 2) for i, rel in enumerate(scores))


def ndcg_at_k(relevance_scores: list[float], k: int) -> float:
    actual = dcg_at_k(relevance_scores, k)
    ideal = dcg_at_k(sorted(relevance_scores, reverse=True), k)
    return actual / ideal if ideal > 0 else 0.0


# ---------------------------------------------------------------------------
# Classification metrics (for the labelled matching dataset)
# ---------------------------------------------------------------------------

def evaluate_classifier(
    y_true: list[str],
    y_pred: list[str],
) -> dict:
    """
    Evaluate label predictions against ground truth.
    Labels: 'No Fit', 'Potential Fit', 'Good Fit'
    """
    true_int = [LABEL_MAP.get(l, 0) for l in y_true]
    pred_int = [LABEL_MAP.get(p, 0) for p in y_pred]

    return {
        "precision_macro": round(precision_score(true_int, pred_int, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(true_int, pred_int, average="macro", zero_division=0), 4),
        "f1_macro": round(f1_score(true_int, pred_int, average="macro", zero_division=0), 4),
        "precision_weighted": round(precision_score(true_int, pred_int, average="weighted", zero_division=0), 4),
        "recall_weighted": round(recall_score(true_int, pred_int, average="weighted", zero_division=0), 4),
        "f1_weighted": round(f1_score(true_int, pred_int, average="weighted", zero_division=0), 4),
        "report": classification_report(true_int, pred_int, target_names=LABEL_names_for(true_int, pred_int), zero_division=0),  # noqa: E501
    }


def LABEL_names_for(true_int, pred_int) -> list[str]:
    all_int = sorted(set(true_int) | set(pred_int))
    rev = {v: k for k, v in LABEL_MAP.items()}
    return [rev.get(i, str(i)) for i in all_int]


def score_to_label(score: float) -> str:
    """Convert a continuous hybrid score [0,1] to a fit label."""
    if score >= 0.65:
        return "Good Fit"
    elif score >= 0.40:
        return "Potential Fit"
    return "No Fit"


# ---------------------------------------------------------------------------
# Full evaluation run on the labelled dataset
# ---------------------------------------------------------------------------

def evaluate_on_dataset(
    matcher,
    test_df: pd.DataFrame,
    top_k: int = 5,
    score_col: str = "hybrid_score",
) -> dict:
    """
    Run the matcher on the labelled test set and compute all metrics.

    Args:
        matcher: A fitted HybridMatcher (or any matcher with .rank()).
        test_df: DataFrame with columns resume_text, job_description_text, label.
        top_k: K for Precision@K, Recall@K, NDCG@K.
        score_col: Column name produced by the matcher to use as the ranking score.

    Returns:
        Dict of aggregate metrics.
    """
    precisions, recalls, mrrs, ndcgs = [], [], [], []
    y_true, y_pred = [], []

    for _, row in test_df.iterrows():
        # We treat each (resume, jd) pair individually:
        # rank the single JD against the resume and record the score.
        query = row["resume_text"]
        label = row["label"]

        # For the labelled dataset, we score the single JD directly
        # using the transformer matcher's encode (cosine similarity).
        jd_text = row["job_description_text"]

        if hasattr(matcher, "transformer"):
            q_emb = matcher.transformer.encode([query])
            jd_emb = matcher.transformer.encode([jd_text])
            score = float((q_emb @ jd_emb.T).flatten()[0])
        else:
            score = 0.5  # fallback

        pred_label = score_to_label(score)
        y_true.append(label)
        y_pred.append(pred_label)

    results = evaluate_classifier(y_true, y_pred)
    results["n_samples"] = len(test_df)
    return results
