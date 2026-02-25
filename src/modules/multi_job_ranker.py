# src/modules/multi_job_ranker.py
"""
MultiJobRanker: Ranks one resume against a pool of jobs.

Pipeline:
  1. Score resume vs each job     (Matcher)
  2. Normalize scores             (ScoreNormalizer)
  3. Compute percentiles          (PercentileCalculator)
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


class MultiJobRanker:
    """
    Ranks one resume against many jobs.

    Accepts a matcher (any object with a .match(resume, job) method
    that returns an object with an .overall_score float attribute).

    Supports two job input formats:
      - Plain dict:    {"job_id": "...", ...}         (mock/unit tests)
      - Wrapped dict:  {"job_id": "...", "profile": JobProfile}
                                                      (integration/real use)

    Usage:
        ranker  = MultiJobRanker(matcher)
        results = ranker.rank(resume_data, job_list)
        # results[0] is the best-matching job
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
        logger.info("MultiJobRanker initialized")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def rank(
        self,
        resume:   Any,
        jobs:     List[Dict],
        id_field: str = "job_id",
    ) -> List[RankingResult]:
        """
        Rank one resume against a list of jobs.

        Args:
            resume:   Resume data (ResumeProfile, dict, or string —
                      whatever your matcher expects)
            jobs:     List of job dicts. Each must have the key
                      specified by id_field. May optionally contain
                      a "profile" key with the actual JobProfile object.
            id_field: Key used to extract the job identifier
                      (default: "job_id")

        Returns:
            List[RankingResult] sorted by overall_score descending,
            with ranks 1, 2, 3... assigned. Returns [] if jobs=[].

        Example:
            >>> results = ranker.rank(resume, jobs)
            >>> print(results[0].rank, results[0].job_id, results[0].overall_score)
            1  job_042  0.87
        """
        if not jobs:
            logger.warning("MultiJobRanker.rank() called with empty job list")
            return []

        # --- Step 1: Score resume vs each job ---
        match_results = []
        raw_scores    = []

        for job in jobs:
            # Support both plain dicts (mock tests) and profile-wrapped dicts
            job_profile = (
                job["profile"]
                if isinstance(job, dict) and "profile" in job
                else job
            )
            match = self.matcher.match(resume, job_profile)
            match_results.append(match)
            raw_scores.append(match.overall_score)

        logger.debug(f"Scored resume against {len(jobs)} jobs. "
                     f"Score range: [{min(raw_scores):.3f}, {max(raw_scores):.3f}]")

        # --- Step 2: Normalize scores ---
        normalized_scores = self.normalizer.normalize(raw_scores)

        # --- Step 3: Compute percentiles ---
        percentiles = self.percentile.calculate(raw_scores)

        # --- Step 4: Build RankingResult objects (unsorted) ---
        unsorted_results = []
        for i, job in enumerate(jobs):
            # Support both dict and object id extraction
            if isinstance(job, dict):
                job_id = job.get(id_field, f"job_{i}")
            else:
                job_id = getattr(job, id_field, f"job_{i}")

            perc = percentiles[i]

            result = RankingResult(
                rank             = 0,           # assigned after sort
                resume_id        = "",           # no single resume_id in multi-job context
                job_id           = job_id,
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

        logger.info(f"Ranked {len(ranked)} jobs. "
                    f"Best: {ranked[0].job_id} ({ranked[0].overall_score:.3f})")

        return ranked

    def rank_with_details(
        self,
        resume:   Any,
        jobs:     List[Dict],
        id_field: str = "job_id",
    ) -> Dict:
        """
        Rank jobs and return full diagnostic details.

        Returns:
            {
              "ranked":       List[RankingResult],   sorted results
              "n":            int,                    pool size
              "best_job":     str,                    top job's id
              "best_score":   float,                  top raw score
              "worst_score":  float,                  bottom raw score
              "score_range":  float,                  max - min
              "mean_score":   float,                  average raw score
            }
        """
        ranked = self.rank(resume, jobs, id_field)

        if not ranked:
            return {
                "ranked":      [],
                "n":           0,
                "best_job":    None,
                "best_score":  0.0,
                "worst_score": 0.0,
                "score_range": 0.0,
                "mean_score":  0.0,
            }

        scores = [r.overall_score for r in ranked]

        return {
            "ranked":      ranked,
            "n":           len(ranked),
            "best_job":    ranked[0].job_id,
            "best_score":  round(max(scores), 4),
            "worst_score": round(min(scores), 4),
            "score_range": round(max(scores) - min(scores), 4),
            "mean_score":  round(sum(scores) / len(scores), 4),
        }
