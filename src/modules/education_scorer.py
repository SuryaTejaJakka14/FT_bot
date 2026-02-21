# src/modules/education_scorer.py
"""
EducationScorer: Scores how well a candidate's education
meets a job's education requirement.

Maps education text → ordinal level → normalized score.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class EducationScorer:
    """
    Scores candidate education against job education requirement.

    Maps education strings to ordinal levels (0-5) then computes
    a normalized score based on the gap between levels.

    Scoring rules:
      - Candidate meets or exceeds requirement  → 1.0
      - No requirement                          → 1.0
      - Unknown candidate education             → 0.3 (partial credit)
      - Each level below requirement            → -0.20 penalty
      - Score floored at 0.0 for known education

    Usage:
        scorer = EducationScorer()
        score = scorer.score(
            candidate_education="Master in Computer Science",
            required_education="Bachelor in Computer Science",
        )
        print(score)   # 1.0 (Master > Bachelor)
    """

    # Ordinal education levels — checked highest first
    # Each tuple: (level, [keywords])
    EDUCATION_LEVELS: List[Tuple[int, List[str]]] = [
        (5, ["phd", "ph.d", "doctorate", "doctoral"]),
        (4, ["master", "msc", "m.s.", "m.s ", " ms ", "mba", "m.eng", "m.tech"]),
        (3, ["bachelor", "bsc", "b.s.", "b.s ", "bs ", "b.eng", "b.tech",
             "undergraduate", "college degree"]),
        (2, ["associate", "a.s.", "a.a."]),          # ← plain "associate" added
        (1, ["high school", "secondary school", "ged", "hsc", "secondary"]),
    ]

    # Penalty per level below requirement
    PENALTY_PER_LEVEL: float = 0.20

    # Partial credit for unknown education (level 0)
    UNKNOWN_SCORE: float = 0.30

    def __init__(self):
        """Initialize EducationScorer."""
        logger.info("EducationScorer initialized")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def score(
        self,
        candidate_education: str,
        required_education: str,
    ) -> float:
        """
        Score candidate education against job requirement.

        Args:
            candidate_education: Education string from ResumeProfile
            required_education:  Education requirement from JobProfile

        Returns:
            Float score 0.0 → 1.0

        Examples:
            >>> scorer.score("PhD in ML", "Bachelor in CS")
            1.0   # PhD > Bachelor
            >>> scorer.score("Bachelor in CS", "Master in CS")
            0.8   # 1 level below
            >>> scorer.score("Bachelor in CS", "PhD in CS")
            0.6   # 2 levels below
            >>> scorer.score("", "Bachelor in CS")
            0.3   # unknown education → partial credit
            >>> scorer.score("Master in CS", "")
            1.0   # no requirement
        """
        # Detect levels
        required_level  = self._detect_level(required_education)
        candidate_level = self._detect_level(candidate_education)

        # Special case: no requirement
        if required_level == 0:
            logger.debug("No education requirement → score = 1.0")
            return 1.0

        # Special case: unknown candidate education
        if candidate_level == 0:
            logger.debug(
                f"Unknown candidate education "
                f"(required level={required_level}) "
                f"→ partial credit {self.UNKNOWN_SCORE}"
            )
            return self.UNKNOWN_SCORE

        # Candidate meets or exceeds requirement
        if candidate_level >= required_level:
            logger.debug(
                f"Candidate level {candidate_level} >= "
                f"required level {required_level} → score = 1.0"
            )
            return 1.0

        # Linear penalty for gap
        gap   = required_level - candidate_level
        score = 1.0 - (gap * self.PENALTY_PER_LEVEL)
        score = max(0.0, round(score, 4))   # floor at 0.0, NOT UNKNOWN_SCORE

        logger.debug(
            f"Education gap: {gap} levels "
            f"(required={required_level}, candidate={candidate_level}) "
            f"→ score={score:.4f}"
        )

        return score

    def score_with_details(
        self,
        candidate_education: str,
        required_education: str,
    ) -> dict:
        """
        Score with full diagnostic details.

        Returns:
            Dictionary with keys:
              - score:             float (final score)
              - candidate_level:   int (detected education level 0-5)
              - required_level:    int (detected education level 0-5)
              - candidate_label:   str (human-readable level name)
              - required_label:    str (human-readable level name)
              - gap_levels:        int (levels below requirement)
              - meets_requirement: bool
        """
        required_level  = self._detect_level(required_education)
        candidate_level = self._detect_level(candidate_education)
        gap             = max(0, required_level - candidate_level)
        final_score     = self.score(candidate_education, required_education)
        meets           = candidate_level >= required_level and required_level > 0

        return {
            "score":             final_score,
            "candidate_level":   candidate_level,
            "required_level":    required_level,
            "candidate_label":   self._level_label(candidate_level),
            "required_label":    self._level_label(required_level),
            "gap_levels":        gap,
            "meets_requirement": meets,
        }

    def get_education_level(self, education_text: str) -> int:
        """
        Public method to get the detected education level.

        Args:
            education_text: Education string to classify

        Returns:
            Integer level 0-5
        """
        return self._detect_level(education_text)

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _detect_level(self, text: str) -> int:
        """
        Detect the education level from a text string.

        Scans from highest level (PhD=5) to lowest (High School=1).
        Returns 0 if no level detected (unknown).

        Args:
            text: Education text to classify

        Returns:
            Integer level 0-5
        """
        if not text or not text.strip():
            return 0

        text_lower = text.lower()

        # Scan highest to lowest — return on first match
        for level, keywords in self.EDUCATION_LEVELS:
            for keyword in keywords:
                if keyword in text_lower:
                    logger.debug(
                        f"Detected level {level} "
                        f"(keyword='{keyword}') in '{text[:40]}'"
                    )
                    return level

        logger.debug(f"No education level detected in '{text[:40]}'")
        return 0

    def _level_label(self, level: int) -> str:
        """
        Return a human-readable label for an education level.

        Args:
            level: Integer level 0-5

        Returns:
            String label
        """
        labels = {
            0: "Unknown",
            1: "High School",
            2: "Associate",
            3: "Bachelor",
            4: "Master",
            5: "PhD",
        }
        return labels.get(level, "Unknown")
