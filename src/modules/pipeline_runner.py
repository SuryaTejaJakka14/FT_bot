# src/modules/pipeline_runner.py
"""
PipelineRunner: End-to-end orchestrator for Module 5.

Chains:
  ResumeParser → PipelineJobScraper → MatchingEngine
  → RankingEngine → min_score filter → JobStore

One .run() call goes from config settings to a saved, ranked
list of jobs in the database.

Usage:
    config = PipelineConfig(
        search_role = "ML Engineer",
        resume_path = "data/resumes/my_resume.pdf",
    )
    runner = PipelineRunner(config)
    result = runner.run()

    print(f"Found {result.jobs_found} jobs")
    print(f"Saved {result.jobs_saved} new matches")
    print(f"Top match: {result.top_match}")
"""

import logging
import time
from dataclasses import dataclass, field
from typing      import List, Optional

from src.modules.pipeline_config  import PipelineConfig
from src.modules.pipeline_scraper import PipelineJobScraper, ScrapedJob
from src.modules.matching_engine  import MatchingEngine
from src.modules.ranking_engine   import RankingEngine
from src.modules.job_store        import JobStore

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# RUN RESULT
# ------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Summary of one PipelineRunner.run() execution.

    Fields:
        jobs_found:   Total job listings scraped from the web
        jobs_matched: Jobs that scored >= config.min_score
        jobs_new:     Jobs not already in the database
        jobs_saved:   Jobs successfully written to database
        top_match:    "Title @ Company" of the #1 ranked result
        run_duration: Total wall-clock seconds for the full run
        ranked:       Full List[RankingResult] for TrackerUI to display
        error:        Set if the run failed before completing
    """
    jobs_found:   int
    jobs_matched: int
    jobs_new:     int
    jobs_saved:   int
    top_match:    str
    run_duration: float
    ranked:       List  = field(default_factory=list)
    error:        Optional[str] = None

    def summary(self) -> str:
        """One-line human-readable run summary."""
        if self.error:
            return f"Run FAILED: {self.error}"
        return (
            f"Found {self.jobs_found} jobs | "
            f"Matched {self.jobs_matched} | "
            f"New {self.jobs_new} | "
            f"Saved {self.jobs_saved} | "
            f"Top: {self.top_match} | "
            f"{self.run_duration:.4f}s"
        )


# ------------------------------------------------------------------
# PIPELINE RUNNER
# ------------------------------------------------------------------

class PipelineRunner:
    """
    End-to-end job search pipeline orchestrator.

    Wires together:
        PipelineJobScraper  → scrape jobs from web
        MatchingEngine      → score each job vs resume
        RankingEngine       → rank + normalise scores
        JobStore            → persist results

    All components are created internally from config.
    The store is kept open for the lifetime of the runner
    so the TUI can also query it.

    Usage:
        runner = PipelineRunner(config)
        result = runner.run()
        jobs   = runner.store.get_all()   # read from tracker
        runner.close()
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialise all pipeline components from config.

        Args:
            config: PipelineConfig with all search + matching settings
        """
        self.config  = config
        self.store   = JobStore(config.db_path)
        self._matcher = MatchingEngine()
        self._ranker  = RankingEngine(self._matcher)
        self._scraper = PipelineJobScraper(config)

        logger.info(
            f"PipelineRunner ready: role='{config.search_role}', "
            f"db='{config.db_path}'"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def run(self) -> RunResult:
        """
        Execute the full pipeline:
          scrape → match → rank → filter → save

        Returns:
            RunResult with counts, top match, duration, and ranked list.
            On critical failure, returns RunResult with error set.
        """
        t_start = time.perf_counter()
        logger.info(
            f"Pipeline run starting: '{self.config.search_role}' "
            f"in '{self.config.location}'"
        )

        # -- Stage 1: Load resume -----------------------------------
        resume = self._load_resume()
        if resume is None:
            return RunResult(
                jobs_found=0, jobs_matched=0, jobs_new=0,
                jobs_saved=0, top_match="—",
                run_duration=time.perf_counter() - t_start,
                error="Could not load resume",
            )

        # -- Stage 2: Scrape jobs ----------------------------------
        scraped = self._scrape_jobs()
        if not scraped:
            return RunResult(
                jobs_found=0, jobs_matched=0, jobs_new=0,
                jobs_saved=0, top_match="—",
                run_duration=time.perf_counter() - t_start,
                error="No jobs returned from scraper",
            )

        # -- Stage 3 + 4: Match + Rank -----------------------------
        ranked = self._match_and_rank(resume, scraped)

        # -- Stage 5: Filter by min_score --------------------------
        qualified = [
            r for r in ranked
            if r.overall_score >= self.config.min_score
        ]
        logger.info(
            f"Qualified (score >= {self.config.min_score}): "
            f"{len(qualified)}/{len(ranked)}"
        )

        # -- Stage 6: Save to database -----------------------------
        jobs_new, jobs_saved = self._save_results(scraped, qualified)

        # -- Build result ------------------------------------------
        top = (
            f"{scraped[0].raw_title} @ {scraped[0].company}"
            if qualified else "—"
        )
        if qualified:
            # Find the ScrapedJob that matches the top RankingResult
            top_result = qualified[0]
            top_scraped = next(
                (s for s in scraped
                 if s.job_id == top_result.job_id),
                None
            )
            if top_scraped:
                top = f"{top_scraped.raw_title} @ {top_scraped.company}"

        duration = time.perf_counter() - t_start
        result   = RunResult(
            jobs_found   = len(scraped),
            jobs_matched = len(qualified),
            jobs_new     = jobs_new,
            jobs_saved   = jobs_saved,
            top_match    = top,
            run_duration = round(duration, 4),
            ranked       = qualified,
        )

        logger.info(f"Pipeline complete: {result.summary()}")
        return result

    def close(self):
        """Close the database connection and scraper."""
        self.store.close()
        self._scraper.close()
        logger.info("PipelineRunner closed")

    # ------------------------------------------------------------------
    # PRIVATE: PIPELINE STAGES
    # ------------------------------------------------------------------

    def _load_resume(self):
        """
        Load and parse the resume file.

        Returns ResumeProfile on success, None on failure.
        """
        try:
            from src.modules.resume_parser import ResumeParser
            parser = ResumeParser()
            resume = parser.parse(self.config.resume_path)
            logger.info(
                f"Resume loaded: {self.config.resume_path} | "
                f"skills={len(resume.hard_skills)}, "
                f"exp={resume.total_experience_years}yrs"
            )
            return resume
        except Exception as e:
            logger.error(f"Failed to load resume: {e}")
            return None

    def _scrape_jobs(self) -> List[ScrapedJob]:
        """
        Scrape job listings using PipelineJobScraper.

        Returns List[ScrapedJob], empty list on failure.
        """
        try:
            jobs = self._scraper.scrape()
            logger.info(f"Scraped {len(jobs)} job listings")
            return jobs
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            return []

    def _match_and_rank(self, resume, scraped: List[ScrapedJob]) -> List:
        """
        Match resume against each job, then rank all results.

        Wraps each individual match in try/except so one bad
        job description never aborts the full run.

        Returns List[RankingResult] sorted by score descending.
        """
        # Build wrapped dicts for RankingEngine
        # (same format used in Module 4 integration tests)
        job_dicts = []
        for s in scraped:
            job_dicts.append({
                "job_id":  s.job_id,
                "profile": s.profile,
            })

        if not job_dicts:
            return []

        try:
            ranked = self._ranker.rank_jobs_for_resume(
                resume, job_dicts
            )
            logger.info(
                f"Ranked {len(ranked)} jobs. "
                f"Best score: {ranked[0].overall_score:.3f}"
                if ranked else "Ranked 0 jobs"
            )
            return ranked
        except Exception as e:
            logger.error(f"Ranking failed: {e}")
            return []

    def _save_results(
        self,
        scraped:   List[ScrapedJob],
        qualified: List,
    ):
        """
        Save qualified ranked jobs to JobStore.

        Matches each RankingResult back to its ScrapedJob by job_id,
        then calls store.save_job().

        Returns:
            (jobs_new, jobs_saved) counts
        """
        # Build a lookup: job_id → ScrapedJob
        scraped_map = {s.job_id: s for s in scraped}

        jobs_new   = 0
        jobs_saved = 0

        for result in qualified:
            scraped_job = scraped_map.get(result.job_id)
            if scraped_job is None:
                logger.warning(
                    f"No ScrapedJob found for job_id: {result.job_id}"
                )
                continue

            already_exists = self.store.job_exists(scraped_job.job_id)

            try:
                inserted = self.store.save_job(scraped_job, result)
                jobs_saved += 1
                if inserted:
                    jobs_new += 1
            except Exception as e:
                logger.warning(
                    f"Failed to save {scraped_job.raw_title}: {e}"
                )

        logger.info(
            f"Saved {jobs_saved} jobs "
            f"({jobs_new} new, {jobs_saved - jobs_new} duplicates skipped)"
        )
        return jobs_new, jobs_saved
