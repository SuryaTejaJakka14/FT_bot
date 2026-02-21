# src/modules/skills_matcher.py
"""
SkillsMatcher: Matches resume skills against job required skills
using both exact matching and semantic (embedding) similarity.

For each required job skill, finds the best matching resume skill
using pre-computed skill embeddings from both profiles.
"""

import logging
import numpy as np
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class SkillsMatcher:
    """
    Matches resume skills against job required/nice-to-have skills.

    Uses a two-tier approach:
      1. Exact string match (score = 1.0)
      2. Semantic match via embedding cosine similarity

    A match is confirmed when similarity >= MATCH_THRESHOLD.

    Usage:
        matcher = SkillsMatcher()
        result = matcher.match(
            resume_skills=["python", "pytorch"],
            job_required_skills=["python", "tensorflow"],
            job_nice_to_have_skills=["docker"],
            resume_skill_embeddings={...},
            job_skill_embeddings={...},
        )
        print(result["skills_score"])      # 0.75
        print(result["matched_skills"])    # ["python", "tensorflow"]
        print(result["missing_skills"])    # []
        print(result["bonus_skills"])      # []
    """

    # Similarity threshold for semantic skill matching
    MATCH_THRESHOLD: float = 0.65

    def __init__(self, match_threshold: float = None):
        """
        Initialize SkillsMatcher.

        Args:
            match_threshold: Override default similarity threshold (0.65).
                             Higher = stricter matching.
        """
        if match_threshold is not None:
            self.MATCH_THRESHOLD = match_threshold
        logger.info(
            f"SkillsMatcher initialized "
            f"(threshold={self.MATCH_THRESHOLD})"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def match(
        self,
        resume_skills: List[str],
        job_required_skills: List[str],
        job_nice_to_have_skills: List[str],
        resume_skill_embeddings: Dict[str, np.ndarray],
        job_skill_embeddings: Dict[str, np.ndarray],
    ) -> dict:
        """
        Match resume skills against job requirements.

        For each required job skill, finds the best matching
        resume skill using exact + semantic matching.

        Args:
            resume_skills:            List of skills from resume
            job_required_skills:      Must-have skills from job
            job_nice_to_have_skills:  Preferred skills from job
            resume_skill_embeddings:  {skill: embedding} from ResumeProfile
            job_skill_embeddings:     {skill: embedding} from JobProfile

        Returns:
            Dictionary with keys:
              - skills_score:       float (0.0 → 1.0)
              - matched_skills:     List[str] required skills matched
              - missing_skills:     List[str] required skills not matched
              - bonus_skills:       List[str] nice-to-have skills matched
              - skill_similarities: Dict[str, float] per-skill best scores

        Example:
            >>> result = matcher.match(
            ...     resume_skills=["python", "pytorch"],
            ...     job_required_skills=["python", "tensorflow"],
            ...     job_nice_to_have_skills=["docker"],
            ...     resume_skill_embeddings={...},
            ...     job_skill_embeddings={...},
            ... )
            >>> print(result["skills_score"])
            1.0   # pytorch ≈ tensorflow above threshold
        """
        if not job_required_skills:
            logger.debug("No required skills in job — skills_score = 1.0")
            return self._empty_result(score=1.0)

        if not resume_skills:
            logger.debug("No skills in resume — all required skills missing")
            return self._empty_result(
                score=0.0,
                missing_skills=list(job_required_skills)
            )

        # Step 1: Match required skills
        matched, missing, similarities = self._match_required_skills(
            resume_skills=resume_skills,
            job_required_skills=job_required_skills,
            resume_skill_embeddings=resume_skill_embeddings,
            job_skill_embeddings=job_skill_embeddings,
        )

        # Step 2: Compute skills score
        skills_score = len(matched) / len(job_required_skills)

        # Step 3: Find bonus skills (nice-to-have matches)
        bonus_skills = self._match_nice_to_have(
            resume_skills=resume_skills,
            job_nice_to_have_skills=job_nice_to_have_skills,
            resume_skill_embeddings=resume_skill_embeddings,
            job_skill_embeddings=job_skill_embeddings,
        )

        logger.debug(
            f"Skills match: {len(matched)}/{len(job_required_skills)} required, "
            f"{len(bonus_skills)} bonus"
        )

        return {
            "skills_score":       round(skills_score, 4),
            "matched_skills":     matched,
            "missing_skills":     missing,
            "bonus_skills":       bonus_skills,
            "skill_similarities": similarities,
        }

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _match_required_skills(
        self,
        resume_skills: List[str],
        job_required_skills: List[str],
        resume_skill_embeddings: Dict[str, np.ndarray],
        job_skill_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[List[str], List[str], Dict[str, float]]:
        """
        For each required job skill, find the best matching resume skill.

        Returns:
            (matched_skills, missing_skills, skill_similarities)
        """
        matched     = []
        missing     = []
        similarities = {}

        for job_skill in job_required_skills:
            best_score, best_resume_skill = self._find_best_match(
                job_skill=job_skill,
                resume_skills=resume_skills,
                resume_skill_embeddings=resume_skill_embeddings,
                job_skill_embeddings=job_skill_embeddings,
            )

            similarities[job_skill] = round(best_score, 4)

            if best_score >= self.MATCH_THRESHOLD:
                matched.append(job_skill)
                logger.debug(
                    f"  MATCH: job='{job_skill}' ↔ "
                    f"resume='{best_resume_skill}' "
                    f"score={best_score:.3f}"
                )
            else:
                missing.append(job_skill)
                logger.debug(
                    f"  MISS:  job='{job_skill}' "
                    f"best_score={best_score:.3f} "
                    f"(below threshold {self.MATCH_THRESHOLD})"
                )

        return matched, missing, similarities

    def _match_nice_to_have(
        self,
        resume_skills: List[str],
        job_nice_to_have_skills: List[str],
        resume_skill_embeddings: Dict[str, np.ndarray],
        job_skill_embeddings: Dict[str, np.ndarray],
    ) -> List[str]:
        """
        Check which nice-to-have skills the candidate has.

        Args:
            resume_skills:           Skills from the resume
            job_nice_to_have_skills: Preferred job skills
            resume_skill_embeddings: Resume skill embeddings
            job_skill_embeddings:    Job skill embeddings

        Returns:
            List of nice-to-have skills the candidate has
        """
        if not job_nice_to_have_skills:
            return []

        bonus = []
        for job_skill in job_nice_to_have_skills:
            best_score, _ = self._find_best_match(
                job_skill=job_skill,
                resume_skills=resume_skills,
                resume_skill_embeddings=resume_skill_embeddings,
                job_skill_embeddings=job_skill_embeddings,
            )
            if best_score >= self.MATCH_THRESHOLD:
                bonus.append(job_skill)

        return bonus

    def _find_best_match(
        self,
        job_skill: str,
        resume_skills: List[str],
        resume_skill_embeddings: Dict[str, np.ndarray],
        job_skill_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[float, str]:
        """
        Find the best matching resume skill for a given job skill.

        Strategy:
          1. Exact string match → return 1.0 immediately
          2. Semantic match via cosine similarity
          3. Return (best_score, best_resume_skill)

        Args:
            job_skill:               Target job skill to match
            resume_skills:           All resume skills
            resume_skill_embeddings: Resume skill embedding dict
            job_skill_embeddings:    Job skill embedding dict

        Returns:
            (best_score, best_resume_skill_name)
        """
        best_score        = 0.0
        best_resume_skill = ""

        # Get job skill embedding (if available)
        job_emb = job_skill_embeddings.get(job_skill)

        for resume_skill in resume_skills:

            # Tier 1: Exact match
            if resume_skill.lower() == job_skill.lower():
                return 1.0, resume_skill

            # Tier 2: Semantic match
            resume_emb = resume_skill_embeddings.get(resume_skill)

            if job_emb is not None and resume_emb is not None:
                score = self._cosine_similarity(resume_emb, job_emb)
            else:
                # Fallback: partial string match
                score = self._partial_string_score(resume_skill, job_skill)

            if score > best_score:
                best_score        = score
                best_resume_skill = resume_skill

        return best_score, best_resume_skill

    def _cosine_similarity(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two vectors."""
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

    def _partial_string_score(
        self,
        skill_a: str,
        skill_b: str,
    ) -> float:
        """
        Fallback scoring when embeddings are unavailable.
        Returns 1.0 for substring match, 0.0 otherwise.

        Examples:
            "apache spark" ↔ "spark" → 1.0
            "python"       ↔ "java"  → 0.0
        """
        a = skill_a.lower()
        b = skill_b.lower()
        if a in b or b in a:
            return 1.0
        return 0.0

    def _empty_result(
        self,
        score: float = 0.0,
        missing_skills: List[str] = None,
    ) -> dict:
        """Return an empty result dictionary."""
        return {
            "skills_score":       score,
            "matched_skills":     [],
            "missing_skills":     missing_skills or [],
            "bonus_skills":       [],
            "skill_similarities": {},
        }
