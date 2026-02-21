# src/modules/experience_scorer.py
"""
ExperienceScorer: Scores how well a candidate's years of experience
meets a job's experience requirement.

Uses a linear penalty curve: each missing year deducts a fixed
amount from the score, floored at 0.0.
"""

import logging

logger = logging.getLogger(__name__)


class ExperienceScorer:
    """
    Scores candidate experience against job experience requirement.

    Scoring rules:
      - Candidate meets or exceeds requirement  → 1.0
      - No requirement (required = 0.0)         → 1.0
      - Each year below requirement             → -PENALTY_PER_YEAR
      - Score is floored at 0.0

    Default penalty: 0.15 per missing year
      → 1 year short  = 0.85
      → 2 years short = 0.70
      → 3 years short = 0.55
      → 7+ years short = 0.00

    Usage:
        scorer = ExperienceScorer()
        score = scorer.score(
            candidate_years=3.0,
            required_years=5.0,
        )
        print(score)   # 0.70
    """

    # Default penalty deducted per missing year
    PENALTY_PER_YEAR: float = 0.15

    def __init__(self, penalty_per_year: float = None):
        """
        Initialize ExperienceScorer.

        Args:
            penalty_per_year: Override default penalty (0.15).
                              Higher = stricter experience matching.
        """
        if penalty_per_year is not None:
            self.PENALTY_PER_YEAR = penalty_per_year
        logger.info(
            f"ExperienceScorer initialized "
            f"(penalty_per_year={self.PENALTY_PER_YEAR})"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def score(
        self,
        candidate_years: float,
        required_years: float,
    ) -> float:
        """
        Score candidate experience against job requirement.

        Args:
            candidate_years: Years of experience from ResumeProfile
            required_years:  Minimum years required from JobProfile

        Returns:
            Float score 0.0 → 1.0

        Examples:
            >>> scorer.score(5.0, 5.0)   # exact match
            1.0
            >>> scorer.score(7.0, 5.0)   # surplus
            1.0
            >>> scorer.score(3.0, 5.0)   # 2 years short
            0.70
            >>> scorer.score(0.0, 5.0)   # no experience
            0.25
            >>> scorer.score(3.0, 0.0)   # no requirement
            1.0
        """
        # Validate inputs
        candidate_years = max(0.0, float(candidate_years or 0.0))
        required_years  = max(0.0, float(required_years  or 0.0))

        # Special case: no requirement
        if required_years == 0.0:
            logger.debug("No experience requirement → score = 1.0")
            return 1.0

        # Special case: candidate meets or exceeds requirement
        if candidate_years >= required_years:
            logger.debug(
                f"Candidate ({candidate_years}y) meets "
                f"requirement ({required_years}y) → score = 1.0"
            )
            return 1.0

        # Linear penalty for gap
        gap   = required_years - candidate_years
        score = 1.0 - (gap * self.PENALTY_PER_YEAR)
        score = max(0.0, round(score, 4))

        logger.debug(
            f"Experience gap: {gap:.1f}y "
            f"(required={required_years}, candidate={candidate_years}) "
            f"→ score={score:.4f}"
        )

        return score

    def score_with_details(
        self,
        candidate_years: float,
        required_years: float,
    ) -> dict:
        """
        Score with full diagnostic details.

        Returns:
            Dictionary with keys:
              - score:            float (final score)
              - candidate_years:  float
              - required_years:   float
              - gap_years:        float (years below requirement, 0 if met)
              - penalty_applied:  float (total penalty deducted)
              - meets_requirement: bool

        Example:
            >>> details = scorer.score_with_details(3.0, 5.0)
            >>> print(details)
            {
                "score": 0.70,
                "candidate_years": 3.0,
                "required_years": 5.0,
                "gap_years": 2.0,
                "penalty_applied": 0.30,
                "meets_requirement": False
            }
        """
        candidate_years  = max(0.0, float(candidate_years or 0.0))
        required_years   = max(0.0, float(required_years  or 0.0))
        gap              = max(0.0, required_years - candidate_years)
        penalty_applied  = gap * self.PENALTY_PER_YEAR
        final_score      = self.score(candidate_years, required_years)
        meets_requirement = candidate_years >= required_years

        return {
            "score":             final_score,
            "candidate_years":   candidate_years,
            "required_years":    required_years,
            "gap_years":         gap,
            "penalty_applied":   round(penalty_applied, 4),
            "meets_requirement": meets_requirement,
        }
