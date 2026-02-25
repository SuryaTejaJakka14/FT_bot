# src/modules/percentile_calculator.py
"""
PercentileCalculator: Computes percentile rank for each score
in a candidate pool.

Percentile = fraction of pool scoring strictly below a given score.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


class PercentileCalculator:
    """
    Computes percentile rankings for a list of scores.

    Percentile formula:
        percentile(x) = count(scores < x) / N

    Ties receive the same percentile (strictly-less-than definition).
    Single candidate always gets percentile = 1.0.
    Empty list returns [].

    Usage:
        calc = PercentileCalculator()
        scores = [0.95, 0.82, 0.74, 0.61, 0.45]
        percentiles = calc.calculate(scores)
        # [0.80, 0.60, 0.40, 0.20, 0.00]
    """

    def __init__(self):
        """Initialize PercentileCalculator."""
        logger.info("PercentileCalculator initialized")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def calculate(self, scores: List[float]) -> List[float]:
        """
        Compute percentile for each score in the pool.

        Percentile = fraction of pool scoring strictly below this score.
        Preserves input order — output[i] is the percentile of scores[i].

        Args:
            scores: List of raw scores (any range, any order)

        Returns:
            List of percentiles [0.0 → 1.0], same length and
            order as input

        Examples:
            >>> calc.calculate([0.95, 0.82, 0.74, 0.61, 0.45])
            [0.8, 0.6, 0.4, 0.2, 0.0]

            >>> calc.calculate([0.82, 0.82, 0.70])
            [0.3333, 0.3333, 0.0]   # tied → same percentile

            >>> calc.calculate([0.75])
            [1.0]   # single candidate

            >>> calc.calculate([])
            []
        """
        if not scores:
            return []

        if len(scores) == 1:
            return [1.0]

        n = len(scores)
        percentiles = []

        for score in scores:
            count_below = sum(1 for s in scores if s < score)
            percentile  = round(count_below / n, 4)
            percentiles.append(percentile)

        logger.debug(
            f"Calculated percentiles for {n} scores: "
            f"range [{min(percentiles):.3f}, {max(percentiles):.3f}]"
        )

        return percentiles

    def calculate_with_details(self, scores: List[float]) -> dict:
        """
        Compute percentiles with full diagnostic details.

        Returns:
            Dictionary with keys:
              - percentiles:    List[float]  percentile per score
              - raw:            List[float]  original scores
              - n:              int          pool size
              - top_score:      float        highest score in pool
              - bottom_score:   float        lowest score in pool
              - has_ties:       bool         True if any scores tied

        Example:
            >>> details = calc.calculate_with_details([0.82, 0.82, 0.70])
            >>> print(details["has_ties"])
            True
            >>> print(details["percentiles"])
            [0.3333, 0.3333, 0.0]
        """
        if not scores:
            return {
                "percentiles":  [],
                "raw":          [],
                "n":            0,
                "top_score":    0.0,
                "bottom_score": 0.0,
                "has_ties":     False,
            }

        percentiles = self.calculate(scores)
        has_ties    = len(set(scores)) < len(scores)

        return {
            "percentiles":  percentiles,
            "raw":          list(scores),
            "n":            len(scores),
            "top_score":    round(max(scores), 4),
            "bottom_score": round(min(scores), 4),
            "has_ties":     has_ties,
        }

    def get_rank_label(self, percentile: float) -> str:
        """
        Convert a percentile to a human-readable rank label.

        Ranges:
            percentile >= 0.90  → "Top 10%"
            percentile >= 0.75  → "Top 25%"
            percentile >= 0.50  → "Top 50%"
            percentile >= 0.25  → "Top 75%"
            percentile <  0.25  → "Bottom 25%"

        Args:
            percentile: Float in [0.0, 1.0]

        Returns:
            String rank label

        Example:
            >>> calc.get_rank_label(0.92)
            'Top 10%'
        """
        if percentile >= 0.90:
            return "Top 10%"
        elif percentile >= 0.75:
            return "Top 25%"
        elif percentile >= 0.50:
            return "Top 50%"
        elif percentile >= 0.25:
            return "Top 75%"
        else:
            return "Bottom 25%"
