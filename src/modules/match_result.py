# src/modules/match_result.py
"""
MatchResult: Output dataclass for the Module 3 Matching Engine.

Stores all scores, diagnostics, and metadata produced by
MatchingEngine.match(resume_profile, job_profile).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict


@dataclass
class MatchResult:
    """
    Complete result of matching a resume against a job description.

    Scores (all 0.0 → 1.0):
        overall_score:    Final weighted score combining all sub-scores
        semantic_score:   Cosine similarity of resume ↔ job embeddings
        skills_score:     Fraction of required skills matched
        experience_score: How well candidate meets experience requirement
        education_score:  How well candidate meets education requirement

    Skill diagnostics:
        matched_skills:     Required skills the candidate has
        missing_skills:     Required skills the candidate lacks
        bonus_skills:       Nice-to-have skills the candidate has
        skill_similarities: Per-skill best match scores {skill: score}

    Meta
        version:    Schema version for forward compatibility
        created_at: Timestamp of when this match was computed
    """

    # ------------------------------------------------------------------
    # SCORES  (all float, 0.0 → 1.0)
    # ------------------------------------------------------------------
    overall_score:    float = 0.0
    semantic_score:   float = 0.0
    skills_score:     float = 0.0
    experience_score: float = 0.0
    education_score:  float = 0.0

    # ------------------------------------------------------------------
    # SKILL DIAGNOSTICS
    # ------------------------------------------------------------------
    matched_skills:     List[str]        = field(default_factory=list)
    missing_skills:     List[str]        = field(default_factory=list)
    bonus_skills:       List[str]        = field(default_factory=list)
    skill_similarities: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # METADATA
    # ------------------------------------------------------------------
    version:    str      = "1.0"
    created_at: datetime = field(default_factory=datetime.now)

    # ------------------------------------------------------------------
    # CONVENIENCE METHODS
    # ------------------------------------------------------------------

    def is_strong_match(self, threshold: float = 0.75) -> bool:
        """
        Returns True if overall_score meets the strong match threshold.

        Args:
            threshold: Minimum score to be considered a strong match.
                       Defaults to 0.75.

        Returns:
            True if overall_score >= threshold

        Example:
            >>> result.is_strong_match()        # default 0.75
            True
            >>> result.is_strong_match(0.90)    # strict threshold
            False
        """
        return self.overall_score >= threshold

    def get_match_label(self) -> str:
        """
        Return a human-readable label for the overall score.

        Score ranges:
            0.80+  → "Excellent Match"
            0.65+  → "Good Match"
            0.50+  → "Partial Match"
            0.35+  → "Weak Match"
            below  → "Poor Match"

        Returns:
            String label for the match quality

        Example:
            >>> result.overall_score = 0.82
            >>> result.get_match_label()
            'Excellent Match'
        """
        if self.overall_score >= 0.80:
            return "Excellent Match"
        elif self.overall_score >= 0.65:
            return "Good Match"
        elif self.overall_score >= 0.50:
            return "Partial Match"
        elif self.overall_score >= 0.35:
            return "Weak Match"
        else:
            return "Poor Match"

    def get_skills_coverage(self) -> float:
        """
        Calculate percentage of required skills covered.

        Returns:
            Float 0.0 → 1.0 representing skills coverage

        Example:
            >>> result.matched_skills = ["python", "sql"]
            >>> result.missing_skills = ["kubernetes"]
            >>> result.get_skills_coverage()
            0.667   # 2 out of 3 required skills
        """
        total = len(self.matched_skills) + len(self.missing_skills)
        if total == 0:
            return 1.0   # no requirements = full coverage
        return len(self.matched_skills) / total

    def summary(self) -> str:
        """
        Return a concise one-line summary of the match result.

        Returns:
            Formatted summary string

        Example:
            >>> print(result.summary())
            '[Good Match] Score: 0.721 | Skills: 5/7 | Exp: 0.80 | Edu: 1.00'
        """
        total_skills = len(self.matched_skills) + len(self.missing_skills)
        return (
            f"[{self.get_match_label()}] "
            f"Score: {self.overall_score:.3f} | "
            f"Skills: {len(self.matched_skills)}/{total_skills} | "
            f"Exp: {self.experience_score:.2f} | "
            f"Edu: {self.education_score:.2f}"
        )
