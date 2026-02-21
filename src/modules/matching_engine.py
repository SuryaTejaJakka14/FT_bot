# src/modules/matching_engine.py
"""
MatchingEngine: Orchestrates all Module 3 components to produce
a MatchResult from a ResumeProfile and JobProfile.

Pipeline:
    ResumeProfile + JobProfile
        → SemanticScorer     (overall embedding similarity)
        → SkillsMatcher      (required + nice-to-have skills)
        → ExperienceScorer   (years of experience gap)
        → EducationScorer    (education level comparison)
        → ScoreAggregator    (weighted final score)
        → MatchResult        (fully populated output)
"""

import logging
from typing import Optional

from src.modules.match_result import MatchResult
from src.modules.semantic_scorer import SemanticScorer
from src.modules.skills_matcher import SkillsMatcher
from src.modules.experience_scorer import ExperienceScorer
from src.modules.education_scorer import EducationScorer
from src.modules.score_aggregator import ScoreAggregator

logger = logging.getLogger(__name__)


class MatchingEngine:
    """
    Orchestrates the full resume-to-job matching pipeline.

    Coordinates:
      - SemanticScorer:    embedding-based overall similarity
      - SkillsMatcher:     skill-level matching (exact + semantic)
      - ExperienceScorer:  years of experience gap scoring
      - EducationScorer:   education level comparison
      - ScoreAggregator:   weighted final score

    Usage:
        engine = MatchingEngine()
        result = engine.match(resume_profile, job_profile)

        print(result.overall_score)       # 0.847
        print(result.matched_skills)      # ["python", "tensorflow"]
        print(result.missing_skills)      # ["kubernetes"]
        print(result.get_match_label())   # "Excellent Match"
    """

    def __init__(
        self,
        match_threshold: float = 0.65,
        weights: Optional[dict] = None,
    ):
        """
        Initialize all matching pipeline components.

        Args:
            match_threshold: Similarity threshold for skill matching (default 0.65)
            weights:         Optional custom score weights dict.
                             Keys: semantic, skills, experience, education.
                             Must sum to 1.0.
        """
        self.semantic_scorer   = SemanticScorer()
        self.skills_matcher    = SkillsMatcher(match_threshold=match_threshold)
        self.experience_scorer = ExperienceScorer()
        self.education_scorer  = EducationScorer()
        self.score_aggregator  = ScoreAggregator(weights=weights)

        logger.info(
            f"MatchingEngine initialized "
            f"(threshold={match_threshold}, "
            f"weights={self.score_aggregator.weights})"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def match(self, resume_profile, job_profile) -> MatchResult:
        """
        Match a resume profile against a job profile.

        This is the single entry point for the entire Module 3 pipeline.

        Args:
            resume_profile: ResumeProfile dataclass from Module 1
            job_profile:    JobProfile dataclass from Module 2

        Returns:
            Fully populated MatchResult dataclass

        Example:
            >>> engine = MatchingEngine()
            >>> result = engine.match(resume_profile, job_profile)
            >>> print(result.overall_score)
            0.847
        """
        logger.info(
            f"Matching resume against job: '{job_profile.title}'"
        )

        # Step 1: Semantic score
        semantic_score = self._compute_semantic_score(
            resume_profile, job_profile
        )

        # Step 2: Skills match
        skills_result = self._compute_skills_match(
            resume_profile, job_profile
        )

        # Step 3: Experience score
        experience_score = self._compute_experience_score(
            resume_profile, job_profile
        )

        # Step 4: Education score
        education_score = self._compute_education_score(
            resume_profile, job_profile
        )

        # Step 5: Aggregate final score
        overall_score = self.score_aggregator.aggregate(
            semantic_score=semantic_score,
            skills_score=skills_result["skills_score"],
            experience_score=experience_score,
            education_score=education_score,
        )

        # Step 6: Assemble MatchResult
        result = MatchResult(
            overall_score=overall_score,
            semantic_score=semantic_score,
            skills_score=skills_result["skills_score"],
            experience_score=experience_score,
            education_score=education_score,
            matched_skills=skills_result["matched_skills"],
            missing_skills=skills_result["missing_skills"],
            bonus_skills=skills_result["bonus_skills"],
            skill_similarities=skills_result["skill_similarities"],
        )

        logger.info(
            f"Match complete — overall={result.overall_score:.3f} "
            f"({result.get_match_label()}) | "
            f"skills={result.skills_score:.3f} | "
            f"semantic={result.semantic_score:.3f}"
        )

        return result

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _compute_semantic_score(
        self, resume_profile, job_profile
    ) -> float:
        """
        Step 1: Compute semantic similarity between embeddings.

        Falls back to 0.0 if either embedding is missing/zero.
        """
        try:
            return self.semantic_scorer.score(
                resume_embedding=resume_profile.resume_embedding,
                job_embedding=job_profile.job_embedding,
            )
        except Exception as e:
            logger.warning(f"Semantic scoring failed: {e} → defaulting to 0.0")
            return 0.0

    def _compute_skills_match(
        self, resume_profile, job_profile
    ) -> dict:
        """
        Step 2: Match resume skills against job required/preferred skills.

        Returns full skills_matcher result dict.
        """
        try:
            return self.skills_matcher.match(
                resume_skills=resume_profile.hard_skills,
                job_required_skills=job_profile.required_hard_skills,
                job_nice_to_have_skills=job_profile.nice_to_have_skills,
                resume_skill_embeddings=resume_profile.skills_embeddings,
                job_skill_embeddings=job_profile.skills_embeddings,
            )
        except Exception as e:
            logger.warning(f"Skills matching failed: {e} → defaulting to empty")
            return {
                "skills_score":       0.0,
                "matched_skills":     [],
                "missing_skills":     list(job_profile.required_hard_skills),
                "bonus_skills":       [],
                "skill_similarities": {},
            }

    def _compute_experience_score(
        self, resume_profile, job_profile
    ) -> float:
        """
        Step 3: Score candidate experience against job requirement.
        """
        try:
            candidate_years = getattr(
                resume_profile, "total_experience_years", 0.0
            )
            return self.experience_scorer.score(
                candidate_years=candidate_years,
                required_years=job_profile.required_experience_years,
            )
        except Exception as e:
            logger.warning(f"Experience scoring failed: {e} → defaulting to 0.0")
            return 0.0

    def _compute_education_score(
        self, resume_profile, job_profile
    ) -> float:
        """
        Step 4: Score candidate education against job requirement.
        """
        try:
            return self.education_scorer.score(
                candidate_education=resume_profile.education,
                required_education=job_profile.required_education,
            )
        except Exception as e:
            logger.warning(f"Education scoring failed: {e} → defaulting to 0.3")
            return 0.3
