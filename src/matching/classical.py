"""
Classical matching baseline: TF-IDF cosine similarity + BM25 ranking.
Both scores are combined into a single classical_score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ClassicalMatcher:
    """
    Two-stage classical matcher:
      1. BM25 for fast first-pass retrieval
      2. TF-IDF cosine similarity for re-scoring
    Final score = alpha * tfidf_score + (1 - alpha) * bm25_score_normalised
    """

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=60_000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.bm25: BM25Okapi | None = None
        self.corpus_df: pd.DataFrame | None = None
        self.tfidf_matrix = None

    def fit(self, corpus_df: pd.DataFrame, text_col: str = "job_description") -> "ClassicalMatcher":
        """Fit vectorisers on the job description corpus."""
        self.corpus_df = corpus_df.reset_index(drop=True)
        texts = corpus_df[text_col].fillna("").astype(str).tolist()

        # TF-IDF
        self.tfidf_matrix = self.tfidf.fit_transform(texts)

        # BM25
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        return self

    def rank(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Rank the corpus against a query string.
        Returns top_k rows from corpus_df with added score columns.
        """
        if self.corpus_df is None or self.tfidf_matrix is None:
            raise RuntimeError("Call fit() before rank().")

        # TF-IDF scores
        q_vec = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # BM25 scores (normalised to [0, 1])
        bm25_raw = np.array(self.bm25.get_scores(query.lower().split()))
        bm25_max = bm25_raw.max() or 1.0
        bm25_scores = bm25_raw / bm25_max

        combined = self.alpha * tfidf_scores + (1 - self.alpha) * bm25_scores
        top_idx = np.argsort(combined)[::-1][:top_k]

        results = self.corpus_df.iloc[top_idx].copy()
        results["tfidf_score"] = np.round(tfidf_scores[top_idx], 4)
        results["bm25_score"] = np.round(bm25_scores[top_idx], 4)
        results["classical_score"] = np.round(combined[top_idx], 4)

        return results.reset_index(drop=True)
