# src/modules/score_normalizer.py
"""
ScoreNormalizer: Applies min-max normalization to a list of scores.

Stretches scores from their natural range [min, max] to [0.0, 1.0]
so small differences between candidates become more visible.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class ScoreNormalizer:
    """
    Normalizes a list of scores using min-max normalization.

    Formula:
        normalized = (x - min) / (max - min)

    Edge cases:
        - All scores equal   → everyone gets 1.0 (tied, all qualified)
        - Single score       → returns 1.0 (no competition)
        - Empty list         → returns []
        - Scores out of range → clamped to [0.0, 1.0] after normalization

    Usage:
        normalizer = ScoreNormalizer()
        raw    = [0.82, 0.78, 0.74, 0.71, 0.68]
        normed = normalizer.normalize(raw)
        # [1.0, 0.7143, 0.4286, 0.2143, 0.0]
    """

    def __init__(self):
        """Initialize ScoreNormalizer."""
        logger.info("ScoreNormalizer initialized")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def normalize(self, scores: List[float]) -> List[float]:
        """
        Apply min-max normalization to a list of scores.

        Preserves the relative order of scores while stretching
        the range to [0.0, 1.0].

        Args:
            scores: List of raw scores (any range)

        Returns:
            List of normalized scores in [0.0, 1.0],
            same length and order as input

        Examples:
            >>> normalizer.normalize([0.82, 0.78, 0.74, 0.68])
            [1.0, 0.7143, 0.4286, 0.0]

            >>> normalizer.normalize([0.75, 0.75, 0.75])
            [1.0, 1.0, 1.0]   # all tied

            >>> normalizer.normalize([0.90])
            [1.0]              # single score

            >>> normalizer.normalize([])
            []                 # empty list
        """
        if not scores:
            return []

        if len(scores) == 1:
            return [1.0]

        min_score = min(scores)
        max_score = max(scores)

        # Edge case: all scores are equal
        if max_score == min_score:
            logger.debug(
                f"All scores equal ({min_score:.4f}) → "
                f"returning 1.0 for all {len(scores)} candidates"
            )
            return [1.0] * len(scores)

        score_range = max_score - min_score
        normalized = [
            round(max(0.0, min(1.0, (s - min_score) / score_range)), 4)
            for s in scores
        ]

        logger.debug(
            f"Normalized {len(scores)} scores: "
            f"range [{min_score:.3f}, {max_score:.3f}] → [0.0, 1.0]"
        )

        return normalized

    def normalize_with_details(self, scores: List[float]) -> dict:
        """
        Normalize scores and return full diagnostic details.

        Returns:
            Dictionary with keys:
              - normalized:   List[float]  normalized scores
              - raw:          List[float]  original scores
              - min_score:    float        minimum raw score
              - max_score:    float        maximum raw score
              - score_range:  float        max - min
              - all_equal:    bool         True if all scores tied

        Example:
            >>> details = normalizer.normalize_with_details([0.82, 0.68])
            >>> print(details["normalized"])
            [1.0, 0.0]
            >>> print(details["score_range"])
            0.14
        """
        if not scores:
            return {
                "normalized":  [],
                "raw":         [],
                "min_score":   0.0,
                "max_score":   0.0,
                "score_range": 0.0,
                "all_equal":   True,
            }

        min_score  = min(scores)
        max_score  = max(scores)
        all_equal  = max_score == min_score
        normalized = self.normalize(scores)

        return {
            "normalized":  normalized,
            "raw":         list(scores),
            "min_score":   round(min_score, 4),
            "max_score":   round(max_score, 4),
            "score_range": round(max_score - min_score, 4),
            "all_equal":   all_equal,
        }

