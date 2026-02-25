# src/modules/multi_resume_ranker.py
"""
MultiResumeRanker: Ranks a pool of resumes against one job.

Pipeline:
  1. Score each resume vs the job   (Matcher)
  2. Normalize scores               (ScoreNormalizer)
  3. Compute percentiles            (PercentileCalculator)
  4. Build RankingResult objects
  5. Sort by score descending
  6. Assign integer ranks 1, 2, 3...
  7. Return ranked list
"""

import logging
from typing import List, Dict, Any

from src.modules.score_normalizer      import ScoreNormalizer
from src.modules.percentile_calculator import PercentileCalculator
from src.modules.ranking_result        import RankingResult

logger = logging.getLogger(__name__)


class MultiResumeRanker:
    """
    Ranks many resumes against one job.

    Accepts a matcher (any object with a .match(resume, job) method
    that returns an object with an .overall_score float attribute).

    Supports two resume input formats:
      - Plain dict:    {"resume_id": "...", ...}      (mock/unit tests)
      - Wrapped dict:  {"resume_id": "...", "profile": ResumeProfile}
                                                      (integration/real use)

    Usage:
        ranker  = MultiResumeRanker(matcher)
        results = ranker.rank(job_data, resume_list)
        # results[0] is the best-matching candidate
    """

    def __init__(self, matcher):
        """
        Args:
            matcher: Any matcher with a .match(resume, job) → MatchResult
                     MatchResult must have an .overall_score float attribute.
        """
        self.matcher    = matcher
        self.normalizer = ScoreNormalizer()
        self.percentile = PercentileCalculator()
        logger.info("MultiResumeRanker initialized")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def rank(
        self,
        job:      Any,
        resumes:  List[Dict],
        id_field: str = "resume_id",
    ) -> List[RankingResult]:
        """
        Rank a list of resumes against one job.

        Args:
            job:      Job data (JobProfile, dict, or string —
                      whatever your matcher expects)
            resumes:  List of resume dicts. Each must have the key
                      specified by id_field. May optionally contain
                      a "profile" key with the actual ResumeProfile object.
            id_field: Key used to extract the resume identifier
                      (default: "resume_id")

        Returns:
            List[RankingResult] sorted by overall_score descending,
            with ranks 1, 2, 3... assigned. Returns [] if resumes=[].

        Example:
            >>> results = ranker.rank(job, resumes)
            >>> print(results[0].rank, results[0].resume_id, results[0].overall_score)
            1  alice  0.91
        """
        if not resumes:
            logger.warning("MultiResumeRanker.rank() called with empty resume list")
            return []

        # --- Step 1: Score each resume vs the job ---
        match_results = []
        raw_scores    = []

        for resume in resumes:
            # Support both plain dicts (mock tests) and profile-wrapped dicts
            resume_profile = (
                resume["profile"]
                if isinstance(resume, dict) and "profile" in resume
                else resume
            )
            match = self.matcher.match(resume_profile, job)
            match_results.append(match)
            raw_scores.append(match.overall_score)

        logger.debug(f"Scored {len(resumes)} resumes against job. "
                     f"Score range: [{min(raw_scores):.3f}, {max(raw_scores):.3f}]")

        # --- Step 2: Normalize scores ---
        normalized_scores = self.normalizer.normalize(raw_scores)

        # --- Step 3: Compute percentiles ---
        percentiles = self.percentile.calculate(raw_scores)

        # --- Step 4: Build RankingResult objects (unsorted) ---
        unsorted_results = []
        for i, resume in enumerate(resumes):
            # Support both dict and object id extraction
            if isinstance(resume, dict):
                resume_id = resume.get(id_field, f"resume_{i}")
            else:
                resume_id = getattr(resume, id_field, f"resume_{i}")

            perc = percentiles[i]

            result = RankingResult(
                rank             = 0,           # assigned after sort
                resume_id        = resume_id,
                job_id           = "",           # no single job_id in multi-resume context
                match_result     = match_results[i],
                percentile       = round(perc, 4),
                normalized_score = round(normalized_scores[i], 4),
            )
            unsorted_results.append(result)

        # --- Step 5 & 6: Sort descending, assign ranks ---
        ranked = sorted(unsorted_results,
                        key=lambda r: r.overall_score,
                        reverse=True)

        for position, result in enumerate(ranked, start=1):
            result.rank = position

        logger.info(f"Ranked {len(ranked)} resumes. "
                    f"Best: {ranked[0].resume_id} ({ranked[0].overall_score:.3f})")

        return ranked

    def rank_with_details(
        self,
        job:      Any,
        resumes:  List[Dict],
        id_field: str = "resume_id",
    ) -> Dict:
        """
        Rank resumes and return full diagnostic details.

        Returns:
            {
              "ranked":          List[RankingResult],  sorted results
              "n":               int,                   pool size
              "best_candidate":  str,                   top resume's id
              "best_score":      float,                 top raw score
              "worst_score":     float,                 bottom raw score
              "score_range":     float,                 max - min
              "mean_score":      float,                 average raw score
            }
        """
        ranked = self.rank(job, resumes, id_field)

        if not ranked:
            return {
                "ranked":         [],
                "n":              0,
                "best_candidate": None,
                "best_score":     0.0,
                "worst_score":    0.0,
                "score_range":    0.0,
                "mean_score":     0.0,
            }

        scores = [r.overall_score for r in ranked]

        return {
            "ranked":         ranked,
            "n":              len(ranked),
            "best_candidate": ranked[0].resume_id,
            "best_score":     round(max(scores), 4),
            "worst_score":    round(min(scores), 4),
            "score_range":    round(max(scores) - min(scores), 4),
            "mean_score":     round(sum(scores) / len(scores), 4),
        }

    def get_shortlist(
        self,
        job:       Any,
        resumes:   List[Dict],
        top_n:     int   = 5,
        min_score: float = 0.0,
        id_field:  str   = "resume_id",
    ) -> List[RankingResult]:
        """
        Rank resumes and return only the top N above a minimum score.

        Args:
            job:       Job data
            resumes:   List of resume dicts
            top_n:     Maximum number of candidates to return (default 5)
            min_score: Minimum overall_score to include (default 0.0)
            id_field:  Key for resume identifier

        Returns:
            List[RankingResult] of length <= top_n,
            all with overall_score >= min_score

        Example:
            >>> shortlist = ranker.get_shortlist(job, resumes, top_n=3, min_score=0.70)
        """
        ranked    = self.rank(job, resumes, id_field)
        shortlist = [r for r in ranked if r.overall_score >= min_score]
        return shortlist[:top_n]
