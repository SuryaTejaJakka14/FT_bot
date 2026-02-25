# src/modules/ranking_engine.py
"""
RankingEngine: Unified facade for all Module 4 ranking operations.

Combines MultiJobRanker and MultiResumeRanker behind a single
clean interface. External code should only import this class.

Usage:
    engine = RankingEngine(matcher)

    # Candidate view: which job fits me best?
    results = engine.rank_jobs_for_resume(resume, jobs)

    # Recruiter view: which candidate fits this job best?
    results = engine.rank_resumes_for_job(job, resumes)

    # Recruiter shortlist: top N qualified candidates
    shortlist = engine.shortlist_resumes(job, resumes, top_n=5, min_score=0.70)
"""

import logging
from typing import List, Dict, Any

from src.modules.multi_job_ranker    import MultiJobRanker
from src.modules.multi_resume_ranker import MultiResumeRanker
from src.modules.ranking_result      import RankingResult

logger = logging.getLogger(__name__)


class RankingEngine:
    """
    Unified entry point for all ranking operations.

    Owns and manages MultiJobRanker and MultiResumeRanker internally.
    Callers never need to import or instantiate those classes directly.

    Args:
        matcher: Any object with a .match(resume, job) → MatchResult.
                 MatchResult must expose an .overall_score float.
    """

    def __init__(self, matcher):
        self.matcher        = matcher
        self._job_ranker    = MultiJobRanker(matcher)
        self._resume_ranker = MultiResumeRanker(matcher)
        logger.info("RankingEngine initialized")

    # ------------------------------------------------------------------
    # CANDIDATE VIEW  (one resume → many jobs)
    # ------------------------------------------------------------------

    def rank_jobs_for_resume(
        self,
        resume:   Any,
        jobs:     List[Dict],
        id_field: str = "job_id",
    ) -> List[RankingResult]:
        """
        Rank a list of jobs for one resume.

        Returns jobs sorted best→worst fit for this resume.

        Args:
            resume:   Resume data (whatever your matcher expects)
            jobs:     List of job dicts, each with id_field key
            id_field: Key for job identifier (default: "job_id")

        Returns:
            List[RankingResult] sorted descending by score.
            results[0] is the best-fit job for this resume.

        Example:
            >>> results = engine.rank_jobs_for_resume(resume, jobs)
            >>> print(results[0].job_id, results[0].overall_score)
            ml_engineer_senior  0.91
        """
        logger.info(f"rank_jobs_for_resume: scoring against {len(jobs)} jobs")
        return self._job_ranker.rank(resume, jobs, id_field)

    # ------------------------------------------------------------------
    # RECRUITER VIEW  (one job → many resumes)
    # ------------------------------------------------------------------

    def rank_resumes_for_job(
        self,
        job:      Any,
        resumes:  List[Dict],
        id_field: str = "resume_id",
    ) -> List[RankingResult]:
        """
        Rank a list of resumes for one job.

        Returns candidates sorted best→worst fit for this job.

        Args:
            job:      Job data (whatever your matcher expects)
            resumes:  List of resume dicts, each with id_field key
            id_field: Key for resume identifier (default: "resume_id")

        Returns:
            List[RankingResult] sorted descending by score.
            results[0] is the best-fit candidate for this job.

        Example:
            >>> results = engine.rank_resumes_for_job(job, resumes)
            >>> print(results[0].resume_id, results[0].overall_score)
            alice  0.91
        """
        logger.info(f"rank_resumes_for_job: scoring {len(resumes)} resumes")
        return self._resume_ranker.rank(job, resumes, id_field)

    def shortlist_resumes(
        self,
        job:       Any,
        resumes:   List[Dict],
        top_n:     int   = 5,
        min_score: float = 0.0,
        id_field:  str   = "resume_id",
    ) -> List[RankingResult]:
        """
        Return the top N candidates above a minimum score threshold.

        Useful for producing an interview shortlist from a large pool.

        Args:
            job:       Job data
            resumes:   List of resume dicts
            top_n:     Maximum candidates to return (default 5)
            min_score: Minimum overall_score to qualify (default 0.0)
            id_field:  Key for resume identifier

        Returns:
            List[RankingResult] of length <= top_n,
            all with overall_score >= min_score.

        Example:
            >>> shortlist = engine.shortlist_resumes(
            ...     job, resumes, top_n=3, min_score=0.75
            ... )
        """
        logger.info(
            f"shortlist_resumes: top_n={top_n}, min_score={min_score}, "
            f"pool={len(resumes)}"
        )
        return self._resume_ranker.get_shortlist(
            job, resumes, top_n, min_score, id_field
        )

    # ------------------------------------------------------------------
    # DIAGNOSTICS
    # ------------------------------------------------------------------

    def get_stats(
        self,
        mode:     str,
        primary:  Any,
        pool:     List[Dict],
        id_field: str = None,
    ) -> Dict:
        """
        Return diagnostic stats for a ranking run without exposing
        internal ranker classes.

        Args:
            mode:     "jobs"    → rank_jobs_for_resume stats
                      "resumes" → rank_resumes_for_job stats
            primary:  Resume (if mode="jobs") or Job (if mode="resumes")
            pool:     List of jobs or resumes to rank against
            id_field: Optional override for identifier key

        Returns:
            Dict with keys: n, best_score, worst_score,
                            score_range, mean_score,
                            best_job / best_candidate (mode-dependent)

        Example:
            >>> stats = engine.get_stats("resumes", job, resumes)
            >>> print(stats["best_candidate"], stats["mean_score"])
            alice  0.698
        """
        if mode == "jobs":
            field = id_field or "job_id"
            return self._job_ranker.rank_with_details(primary, pool, field)
        elif mode == "resumes":
            field = id_field or "resume_id"
            return self._resume_ranker.rank_with_details(primary, pool, field)
        else:
            raise ValueError(f"mode must be 'jobs' or 'resumes', got: '{mode}'")
