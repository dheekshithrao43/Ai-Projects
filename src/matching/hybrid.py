"""
Hybrid matching pipeline:
  Stage 1 — Classical (TF-IDF + BM25) retrieves top N candidates fast.
  Stage 2 — SBERT re-ranks those candidates for semantic precision.
  Final score = w_classical * classical_norm + w_transformer * transformer_score
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.matching.classical import ClassicalMatcher
from src.matching.transformer import TransformerMatcher


class HybridMatcher:
    """
    Two-stage hybrid matcher.
    Classical weight is kept low (0.3) by default because SBERT gives
    much better semantic matching for free-text CVs vs job descriptions.
    """

    def __init__(
        self,
        classical_weight: float = 0.3,
        transformer_weight: float = 0.7,
        first_stage_k: int = 100,
    ):
        self.w_classical = classical_weight
        self.w_transformer = transformer_weight
        self.first_stage_k = first_stage_k

        self.classical = ClassicalMatcher()
        self.transformer = TransformerMatcher()
        self.corpus_df: pd.DataFrame | None = None
        self._fitted = False

    def fit(
        self,
        corpus_df: pd.DataFrame,
        text_col: str = "job_description",
    ) -> "HybridMatcher":
        """Fit both matchers on the job description corpus."""
        self.corpus_df = corpus_df.reset_index(drop=True)

        print("Fitting classical matcher (TF-IDF + BM25)...")
        self.classical.fit(corpus_df, text_col)

        print("Fitting transformer matcher (SBERT)...")
        self.transformer.fit(corpus_df, text_col, show_progress=True)

        self._fitted = True
        return self

    def rank(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Two-stage ranking:
          1. Classical retrieves self.first_stage_k candidates.
          2. SBERT re-ranks those candidates.
        Returns top_k results with tfidf_score, bm25_score,
        classical_score, transformer_score, hybrid_score columns.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before rank().")

        # Stage 1: Classical retrieval
        candidates = self.classical.rank(query, top_k=self.first_stage_k)

        # Stage 2: Transformer re-ranking on the candidate set
        candidates = self.transformer.rank_candidates(query, candidates)

        # Normalise classical score to [0, 1]
        c_scores = candidates["classical_score"].values
        c_max = c_scores.max() or 1.0
        c_norm = c_scores / c_max

        t_scores = candidates["transformer_score"].values

        hybrid = self.w_classical * c_norm + self.w_transformer * t_scores

        candidates = candidates.copy()
        candidates["hybrid_score"] = np.round(hybrid, 4)
        candidates = candidates.sort_values("hybrid_score", ascending=False).head(top_k)

        return candidates.reset_index(drop=True)

    def rank_batch(
        self,
        queries: list[str],
        top_k: int = 5,
    ) -> list[pd.DataFrame]:
        """Run rank() for multiple queries. Returns a list of result DataFrames."""
        return [self.rank(q, top_k=top_k) for q in queries]
