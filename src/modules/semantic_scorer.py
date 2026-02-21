# src/modules/semantic_scorer.py
"""
SemanticScorer: Computes semantic similarity between a resume
and a job description using their pre-computed embeddings.

Uses cosine similarity between the resume_embedding and
job_embedding vectors produced in Modules 1 and 2.
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticScorer:
    """
    Computes overall semantic similarity between resume and job embeddings.

    Uses cosine similarity on the 384-dimensional embedding vectors
    produced by EmbeddingsCreator (all-MiniLM-L6-v2 model).

    Score range: 0.0 (no match) → 1.0 (perfect match)

    Usage:
        scorer = SemanticScorer()
        score = scorer.score(resume_embedding, job_embedding)
        print(score)   # e.g. 0.823
    """

    def __init__(self, min_score: float = 0.0, max_score: float = 1.0):
        """
        Initialize SemanticScorer.

        Args:
            min_score: Floor for returned scores (default 0.0)
            max_score: Ceiling for returned scores (default 1.0)
        """
        self.min_score = min_score
        self.max_score = max_score
        logger.info("SemanticScorer initialized")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def score(
        self,
        resume_embedding: np.ndarray,
        job_embedding: np.ndarray,
    ) -> float:
        """
        Compute semantic similarity between resume and job embeddings.

        Args:
            resume_embedding: Resume embedding vector, shape (384,)
            job_embedding:    Job embedding vector, shape (384,)

        Returns:
            Float score 0.0 → 1.0

        Raises:
            ValueError: If embeddings are empty or shape mismatch

        Example:
            >>> scorer = SemanticScorer()
            >>> score = scorer.score(resume_emb, job_emb)
            >>> print(score)
            0.823
        """
        # Validate inputs
        self._validate_embeddings(resume_embedding, job_embedding)

        # Compute cosine similarity
        raw_score = self._cosine_similarity(resume_embedding, job_embedding)

        # Clamp to [min_score, max_score]
        final_score = float(np.clip(raw_score, self.min_score, self.max_score))

        logger.debug(f"Semantic score: {final_score:.4f}")
        return final_score

    def score_with_details(
        self,
        resume_embedding: np.ndarray,
        job_embedding: np.ndarray,
    ) -> dict:
        """
        Compute semantic similarity and return full details.

        Returns a dictionary with:
          - "score":         float  (final clamped score)
          - "raw_score":     float  (raw cosine similarity)
          - "resume_norm":   float  (L2 norm of resume embedding)
          - "job_norm":      float  (L2 norm of job embedding)
          - "dot_product":   float  (raw dot product)

        Args:
            resume_embedding: Resume embedding vector
            job_embedding:    Job embedding vector

        Returns:
            Dictionary with score and diagnostic details

        Example:
            >>> details = scorer.score_with_details(resume_emb, job_emb)
            >>> print(details["score"])
            0.823
            >>> print(details["resume_norm"])
            1.000
        """
        self._validate_embeddings(resume_embedding, job_embedding)

        resume_norm  = float(np.linalg.norm(resume_embedding))
        job_norm     = float(np.linalg.norm(job_embedding))
        dot_product  = float(np.dot(resume_embedding, job_embedding))
        raw_score    = dot_product / (resume_norm * job_norm + 1e-8)
        final_score  = float(np.clip(raw_score, self.min_score, self.max_score))

        return {
            "score":       final_score,
            "raw_score":   raw_score,
            "resume_norm": resume_norm,
            "job_norm":    job_norm,
            "dot_product": dot_product,
        }

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two vectors.

        Formula: cos(θ) = (A · B) / (||A|| × ||B||)

        The 1e-8 epsilon prevents division by zero for zero vectors.

        Args:
            vec_a: First vector
            vec_b: Second vector

        Returns:
            Cosine similarity float
        """
        dot_product = np.dot(vec_a, vec_b)
        norm_a      = np.linalg.norm(vec_a)
        norm_b      = np.linalg.norm(vec_b)
        return dot_product / (norm_a * norm_b + 1e-8)

    def _validate_embeddings(
        self,
        resume_embedding: np.ndarray,
        job_embedding: np.ndarray,
    ) -> None:
        """
        Validate that embeddings are non-empty numpy arrays
        with matching shapes.

        Args:
            resume_embedding: Resume embedding to validate
            job_embedding:    Job embedding to validate

        Raises:
            ValueError: If either embedding is None, empty,
                        or shapes don't match
        """
        if resume_embedding is None or job_embedding is None:
            raise ValueError("Embeddings cannot be None")

        if resume_embedding.size == 0 or job_embedding.size == 0:
            raise ValueError("Embeddings cannot be empty")

        if resume_embedding.shape != job_embedding.shape:
            raise ValueError(
                f"Embedding shape mismatch: "
                f"resume={resume_embedding.shape}, "
                f"job={job_embedding.shape}"
            )
