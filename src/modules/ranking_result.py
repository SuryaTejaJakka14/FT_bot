# src/modules/ranking_result.py
"""
RankingResult: Output dataclass for the Module 4 Ranking Engine.

Wraps a MatchResult with ranking metadata — position, percentile,
and human-readable rank label — computed relative to a candidate pool.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from src.modules.match_result import MatchResult


@dataclass
class RankingResult:
    """
    Complete ranking result for one resume-job pair.

    Wraps a MatchResult (Module 3) with ranking context:
      - rank:             position in sorted leaderboard (1 = best)
      - percentile:       fraction of pool scoring below this candidate
      - normalized_score: score scaled relative to pool min/max
      - rank_label:       human-readable percentile label

    Also stores identifiers so results can be traced back:
      - resume_id:        identifier for the candidate
      - job_id:           identifier for the job

    Usage:
        result = RankingResult(
            rank=1,
            resume_id="alice",
            job_id="ml_engineer",
            match_result=match_result,
            percentile=0.90,
            normalized_score=0.95,
        )
        print(result.overall_score)      # delegates to match_result
        print(result.rank_label)         # "Top 10%"
        print(result.summary())
    """

    # ------------------------------------------------------------------
    # RANKING METADATA
    # ------------------------------------------------------------------
    rank:             int   = 0      # 1-based position (1 = best match)
    resume_id:        str   = ""     # candidate identifier
    job_id:           str   = ""     # job identifier

    # ------------------------------------------------------------------
    # SCORES
    # ------------------------------------------------------------------
    match_result:     Optional[MatchResult] = None   # full Module 3 result
    percentile:       float = 0.0    # fraction of pool scoring below (0→1)
    normalized_score: float = 0.0    # min-max normalized score (0→1)

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------
    version:    str      = "1.0"
    created_at: datetime = field(default_factory=datetime.now)

    # ------------------------------------------------------------------
    # CONVENIENCE PROPERTIES
    # ------------------------------------------------------------------

    @property
    def overall_score(self) -> float:
        """Delegate to match_result.overall_score."""
        if self.match_result is None:
            return 0.0
        return self.match_result.overall_score

    @property
    def skills_score(self) -> float:
        """Delegate to match_result.skills_score."""
        if self.match_result is None:
            return 0.0
        return self.match_result.skills_score

    @property
    def matched_skills(self) -> list:
        """Delegate to match_result.matched_skills."""
        if self.match_result is None:
            return []
        return self.match_result.matched_skills

    @property
    def missing_skills(self) -> list:
        """Delegate to match_result.missing_skills."""
        if self.match_result is None:
            return []
        return self.match_result.missing_skills

    @property
    def bonus_skills(self) -> list:
        """Delegate to match_result.bonus_skills."""
        if self.match_result is None:
            return []
        return self.match_result.bonus_skills

    # ------------------------------------------------------------------
    # CONVENIENCE METHODS
    # ------------------------------------------------------------------

    @property
    def rank_label(self) -> str:
        """
        Return a human-readable percentile label.

        Ranges:
            percentile >= 0.90  → "Top 10%"
            percentile >= 0.75  → "Top 25%"
            percentile >= 0.50  → "Top 50%"
            percentile >= 0.25  → "Top 75%"
            percentile <  0.25  → "Bottom 25%"

        Returns:
            String rank label

        Example:
            >>> result.percentile = 0.92
            >>> result.rank_label
            'Top 10%'
        """
        if self.percentile >= 0.90:
            return "Top 10%"
        elif self.percentile >= 0.75:
            return "Top 25%"
        elif self.percentile >= 0.50:
            return "Top 50%"
        elif self.percentile >= 0.25:
            return "Top 75%"
        else:
            return "Bottom 25%"

    def get_match_label(self) -> str:
        """
        Delegate match quality label to MatchResult.

        Returns:
            String label from MatchResult.get_match_label()

        Example:
            >>> result.get_match_label()
            'Excellent Match'
        """
        if self.match_result is None:
            return "No Match Data"
        return self.match_result.get_match_label()

    def summary(self) -> str:
        """
        Return a concise one-line summary of the ranking result.

        Format:
            #1 | alice | Score: 0.950 | Top 10% | Excellent Match

        Returns:
            Formatted summary string

        Example:
            >>> print(result.summary())
            '#1 | alice | Score: 0.950 | Top 10% | Excellent Match'
        """
        return (
            f"#{self.rank} | "
            f"{self.resume_id or self.job_id} | "
            f"Score: {self.overall_score:.3f} | "
            f"{self.rank_label} | "
            f"{self.get_match_label()}"
        )
