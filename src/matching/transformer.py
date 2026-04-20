"""
Transformer-based matching using SBERT sentence embeddings.
Model: all-MiniLM-L6-v2 (fast, high quality, 384-dim embeddings)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TransformerMatcher:
    """
    Encodes job descriptions into dense SBERT embeddings.
    At query time, encodes the CV and computes cosine similarity
    (normalised embeddings, so dot product = cosine).
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.corpus_embeddings: np.ndarray | None = None
        self.corpus_df: pd.DataFrame | None = None

    def encode(
        self,
        texts: list[str],
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Encode texts into unit-normalised embeddings."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def fit(
        self,
        corpus_df: pd.DataFrame,
        text_col: str = "job_description",
        batch_size: int = 64,
        show_progress: bool = True,
    ) -> "TransformerMatcher":
        """Pre-encode the job description corpus."""
        self.corpus_df = corpus_df.reset_index(drop=True)
        texts = corpus_df[text_col].fillna("").astype(str).tolist()

        if show_progress:
            print(f"Encoding {len(texts):,} job descriptions with SBERT...")
        self.corpus_embeddings = self.encode(texts, batch_size=batch_size, show_progress=show_progress)
        return self

    def rank(self, query: str, top_k: int = 10) -> pd.DataFrame:
        """
        Rank the corpus against a query (CV text or job description).
        Returns top_k rows with added transformer_score column.
        """
        if self.corpus_df is None or self.corpus_embeddings is None:
            raise RuntimeError("Call fit() before rank().")

        query_emb = self.encode([query])
        # Shape: (1, dim) @ (dim, N) = (1, N)
        scores = (query_emb @ self.corpus_embeddings.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = self.corpus_df.iloc[top_idx].copy()
        results["transformer_score"] = np.round(scores[top_idx], 4)
        return results.reset_index(drop=True)

    def rank_candidates(
        self,
        query: str,
        candidate_df: pd.DataFrame,
        text_col: str = "job_description",
    ) -> pd.DataFrame:
        """
        Re-rank a small set of pre-filtered candidates.
        Used in the hybrid matcher's second stage.
        """
        texts = candidate_df[text_col].fillna("").astype(str).tolist()
        query_emb = self.encode([query])
        cand_embs = self.encode(texts)
        scores = (query_emb @ cand_embs.T).flatten()

        result = candidate_df.copy()
        result["transformer_score"] = np.round(scores, 4)
        return result
