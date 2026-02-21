# src/modules/score_aggregator.py
"""
ScoreAggregator: Combines semantic, skills, experience, and education
sub-scores into a single weighted final score.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)


class ScoreAggregator:
    """
    Combines four sub-scores into a single weighted overall score.

    Default weights:
        semantic:   0.30  (overall semantic similarity)
        skills:     0.40  (technical skills match)
        experience: 0.20  (years of experience)
        education:  0.10  (education level)

    All weights must sum to 1.0.

    Usage:
        aggregator = ScoreAggregator()
        overall = aggregator.aggregate(
            semantic_score=0.82,
            skills_score=0.90,
            experience_score=1.00,
            education_score=1.00,
        )
        print(overall)   # 0.916
    """

    # Default weights — must sum to 1.0
    DEFAULT_WEIGHTS: Dict[str, float] = {
        "semantic":   0.30,
        "skills":     0.40,
        "experience": 0.20,
        "education":  0.10,
    }

    def __init__(self, weights: Dict[str, float] = None):
        """
        Initialize ScoreAggregator with optional custom weights.

        Args:
            weights: Optional dict overriding default weights.
                     Must contain keys: semantic, skills,
                     experience, education.
                     Must sum to 1.0.

        Raises:
            ValueError: If weights don't sum to 1.0
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self._validate_weights()
        logger.info(f"ScoreAggregator initialized (weights={self.weights})")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def aggregate(
        self,
        semantic_score:   float,
        skills_score:     float,
        experience_score: float,
        education_score:  float,
    ) -> float:
        """
        Compute weighted overall score from four sub-scores.

        Formula:
            overall = (semantic   × w_semantic)
                    + (skills     × w_skills)
                    + (experience × w_experience)
                    + (education  × w_education)

        Args:
            semantic_score:   Cosine similarity score (0.0 → 1.0)
            skills_score:     Skills match score (0.0 → 1.0)
            experience_score: Experience gap score (0.0 → 1.0)
            education_score:  Education level score (0.0 → 1.0)

        Returns:
            Weighted overall score (0.0 → 1.0)

        Example:
            >>> aggregator.aggregate(0.82, 0.90, 1.00, 1.00)
            0.916
        """
        # Clamp all inputs to [0.0, 1.0]
        semantic   = max(0.0, min(1.0, float(semantic_score   or 0.0)))
        skills     = max(0.0, min(1.0, float(skills_score     or 0.0)))
        experience = max(0.0, min(1.0, float(experience_score or 0.0)))
        education  = max(0.0, min(1.0, float(education_score  or 0.0)))

        overall = (
            semantic   * self.weights["semantic"]   +
            skills     * self.weights["skills"]     +
            experience * self.weights["experience"] +
            education  * self.weights["education"]
        )

        overall = round(max(0.0, min(1.0, overall)), 4)

        logger.debug(
            f"Aggregated score: {overall:.4f} "
            f"(semantic={semantic:.3f}, skills={skills:.3f}, "
            f"experience={experience:.3f}, education={education:.3f})"
        )

        return overall

    def aggregate_with_details(
        self,
        semantic_score:   float,
        skills_score:     float,
        experience_score: float,
        education_score:  float,
    ) -> dict:
        """
        Compute weighted score and return full breakdown.

        Returns:
            Dictionary with keys:
              - overall_score:       float (final weighted score)
              - semantic_score:      float (input, clamped)
              - skills_score:        float (input, clamped)
              - experience_score:    float (input, clamped)
              - education_score:     float (input, clamped)
              - weighted_semantic:   float (semantic × weight)
              - weighted_skills:     float (skills × weight)
              - weighted_experience: float (experience × weight)
              - weighted_education:  float (education × weight)
              - weights:             dict  (weights used)

        Example:
            >>> details = aggregator.aggregate_with_details(
            ...     0.82, 0.90, 1.00, 1.00
            ... )
            >>> print(details["overall_score"])
            0.916
            >>> print(details["weighted_skills"])
            0.36
        """
        semantic   = max(0.0, min(1.0, float(semantic_score   or 0.0)))
        skills     = max(0.0, min(1.0, float(skills_score     or 0.0)))
        experience = max(0.0, min(1.0, float(experience_score or 0.0)))
        education  = max(0.0, min(1.0, float(education_score  or 0.0)))

        w_semantic   = semantic   * self.weights["semantic"]
        w_skills     = skills     * self.weights["skills"]
        w_experience = experience * self.weights["experience"]
        w_education  = education  * self.weights["education"]

        overall = round(
            max(0.0, min(1.0, w_semantic + w_skills + w_experience + w_education)),
            4
        )

        return {
            "overall_score":       overall,
            "semantic_score":      semantic,
            "skills_score":        skills,
            "experience_score":    experience,
            "education_score":     education,
            "weighted_semantic":   round(w_semantic,   4),
            "weighted_skills":     round(w_skills,     4),
            "weighted_experience": round(w_experience, 4),
            "weighted_education":  round(w_education,  4),
            "weights":             self.weights.copy(),
        }

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _validate_weights(self) -> None:
        """
        Validate that weights contain correct keys and sum to 1.0.

        Raises:
            ValueError: If keys are missing or sum != 1.0
        """
        required_keys = {"semantic", "skills", "experience", "education"}
        missing = required_keys - set(self.weights.keys())
        if missing:
            raise ValueError(f"Missing weight keys: {missing}")

        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f}. "
                f"Weights: {self.weights}"
            )
