# tests/test_pipeline_runner.py
"""
Tests for PipelineRunner.

Uses:
  - MockResumeParser    so no real PDF needed
  - MockScraper         so no network calls
  - Real MatchingEngine + RankingEngine (same as Module 4 integration)
  - Real JobStore       on :memory:
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from datetime  import datetime
from unittest.mock import patch, MagicMock

from src.modules.pipeline_config  import PipelineConfig
from src.modules.pipeline_runner  import PipelineRunner, RunResult
from src.modules.pipeline_scraper import ScrapedJob
from src.modules.resume_parser    import ResumeProfile
from src.modules.job_parser       import JobProfile
from src.modules.job_store        import JobStore


def check(passed: bool, label: str):
    print(f"   {'✓' if passed else '⚠'} {label}")


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

EMBEDDING_DIM = 384

def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed % (2**31))
    v   = rng.randn(EMBEDDING_DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def make_resume_profile() -> ResumeProfile:
    skills = ["python", "machine learning", "tensorflow", "sql", "pandas"]
    return ResumeProfile(
        version                = "1.0",
        hard_skills            = skills,
        soft_skills            = [],
        education              = "Bachelor's in CS",
        total_experience_years = 4.0,
        job_history            = ["ML Engineer at ACME (4 yrs)"],
        resume_embedding       = _unit_vec(1),
        skills_embeddings      = {s: _unit_vec(i+2) for i,s in enumerate(skills)},
        raw_text               = " ".join(skills),
        created_at             = datetime.now(),
    )


def make_scraped_job(
    job_id:   str,
    title:    str,
    required: list,
    seed:     int = 100,
) -> ScrapedJob:
    profile = JobProfile(
        version                   = "1.0",
        title                     = title,
        company                   = "TestCo",
        location                  = "Remote",
        required_hard_skills      = required,
        nice_to_have_skills       = [],
        required_experience_years = 2.0,
        required_education        = "",
        job_embedding             = _unit_vec(seed),
        skills_embeddings         = {s: _unit_vec(seed+i+1)
                                     for i, s in enumerate(required)},
        created_at                = datetime.now(),
    )
    return ScrapedJob(
        profile     = profile,
        job_id      = job_id,
        url         = f"https://example.com/{job_id}",
        date_posted = "2026-02-24",
        source      = "linkedin",
        raw_title   = title,
        company     = "TestCo",
        location    = "Remote",
    )


MOCK_RESUME = make_resume_profile()

MOCK_SCRAPED = [
    make_scraped_job("job001", "ML Engineer",
                     ["python", "machine learning", "tensorflow"], seed=101),
    make_scraped_job("job002", "Data Scientist",
                     ["python", "sql", "pandas"],                  seed=102),
    make_scraped_job("job003", "Backend Engineer",
                     ["python", "docker", "postgres"],             seed=103),
    make_scraped_job("job004", "DevOps Engineer",
                     ["docker", "kubernetes", "terraform"],        seed=104),
]


def make_runner(min_score: float = 0.30) -> PipelineRunner:
    """Create a PipelineRunner wired to :memory: db."""
    config = PipelineConfig(
        search_role = "ML Engineer",
        resume_path = "data/resumes/my_resume.pdf",
        min_score   = min_score,
        db_path     = ":memory:",
    )
    runner = PipelineRunner(config)
    # Replace the store with one we control
    runner.store = JobStore(":memory:")
    return runner


def run_with_mocks(runner: PipelineRunner) -> RunResult:
    """Run pipeline with mocked resume loading and scraping."""
    with patch.object(runner, "_load_resume",  return_value=MOCK_RESUME), \
         patch.object(runner, "_scrape_jobs",  return_value=MOCK_SCRAPED):
        return runner.run()


# ------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TESTING PIPELINE RUNNER")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Successful run — RunResult structure
    # ----------------------------------------------------------------
    print("\n1. Testing successful run — RunResult structure...")

    runner = make_runner(min_score=0.30)
    result = run_with_mocks(runner)

    print(f"\n   {result.summary()}\n")
    print(f"   jobs_found   = {result.jobs_found}")
    print(f"   jobs_matched = {result.jobs_matched}")
    print(f"   jobs_new     = {result.jobs_new}")
    print(f"   jobs_saved   = {result.jobs_saved}")
    print(f"   top_match    = {result.top_match}")
    print(f"   run_duration = {result.run_duration}s")
    print(f"   ranked count = {len(result.ranked)}\n")

    checks = [
        (result.error        is None,      "No error"),
        (result.jobs_found   == 4,         "jobs_found = 4"),
        (result.jobs_matched  > 0,         "jobs_matched > 0"),
        (result.jobs_new      > 0,         "jobs_new > 0"),
        (result.jobs_saved   == result.jobs_matched,
                                           "jobs_saved == jobs_matched"),
        (result.top_match    != "—",       "top_match is set"),
        (result.run_duration  > 0,         "run_duration > 0"),
        (len(result.ranked)  > 0,          "ranked list non-empty"),
    ]
    for passed, label in checks:
        check(passed, label)

    runner.close()

    # ----------------------------------------------------------------
    # 2. Results are saved correctly in JobStore
    # ----------------------------------------------------------------
    print("\n2. Testing JobStore is populated after run...")

    runner2 = make_runner(min_score=0.30)
    result2 = run_with_mocks(runner2)

    all_jobs = runner2.store.get_all()
    stats    = runner2.store.get_stats()

    print(f"\n   Jobs in store: {len(all_jobs)}")
    print(f"   Stats: {stats}\n")

    checks = [
        (len(all_jobs)          == result2.jobs_saved, "Store count == jobs_saved"),
        (stats["total"]         == result2.jobs_saved, "get_stats() total matches"),
        (stats["not_applied"]   == result2.jobs_saved, "All new jobs are NOT_APPLIED"),
        (stats["best_score"]     > 0.0,                "best_score > 0"),
        (all("title"    in j for j in all_jobs),       "All jobs have title"),
        (all("job_id"   in j for j in all_jobs),       "All jobs have job_id"),
        (all("url"      in j for j in all_jobs),       "All jobs have url"),
    ]
    for passed, label in checks:
        check(passed, label)

    runner2.close()

    # ----------------------------------------------------------------
    # 3. min_score filter is respected
    # ----------------------------------------------------------------
    print("\n3. Testing min_score filter...")

    runner_low  = make_runner(min_score=0.0)
    runner_high = make_runner(min_score=0.99)

    result_low  = run_with_mocks(runner_low)
    result_high = run_with_mocks(runner_high)

    print(f"\n   min_score=0.00 → jobs_matched = {result_low.jobs_matched}")
    print(f"   min_score=0.99 → jobs_matched = {result_high.jobs_matched}\n")

    checks = [
        (result_low.jobs_matched  == 4,  "min_score=0.0  → all 4 jobs pass"),
        (result_high.jobs_matched == 0,  "min_score=0.99 → 0 jobs pass"),
    ]
    for passed, label in checks:
        check(passed, label)

    runner_low.close()
    runner_high.close()

    # ----------------------------------------------------------------
    # 4. Duplicate runs — second run saves 0 new jobs
    # ----------------------------------------------------------------
    print("\n4. Testing duplicate detection across runs...")

    runner3 = make_runner(min_score=0.30)
    result_run1 = run_with_mocks(runner3)
    result_run2 = run_with_mocks(runner3)   # same data, same store

    print(f"\n   Run 1: new={result_run1.jobs_new}, saved={result_run1.jobs_saved}")
    print(f"   Run 2: new={result_run2.jobs_new}, saved={result_run2.jobs_saved}\n")

    checks = [
        (result_run1.jobs_new  == result_run1.jobs_saved,
            "Run 1: all jobs are new"),
        (result_run2.jobs_new  == 0,
            "Run 2: 0 new jobs (all duplicates)"),
        (result_run2.jobs_saved == result_run2.jobs_matched,
            "Run 2: matched jobs are still saved (counted)"),
    ]
    for passed, label in checks:
        check(passed, label)

    runner3.close()

    # ----------------------------------------------------------------
    # 5. Resume load failure → graceful RunResult with error
    # ----------------------------------------------------------------
    print("\n5. Testing graceful failure when resume fails to load...")

    runner4 = make_runner()
    with patch.object(runner4, "_load_resume", return_value=None):
        result_fail = runner4.run()

    print(f"\n   error = '{result_fail.error}'\n")

    checks = [
        (result_fail.error       is not None, "error is set"),
        (result_fail.jobs_found  == 0,        "jobs_found = 0"),
        (result_fail.jobs_saved  == 0,        "jobs_saved = 0"),
        (result_fail.top_match   == "—",      "top_match = '—'"),
    ]
    for passed, label in checks:
        check(passed, label)

    runner4.close()

    # ----------------------------------------------------------------
    # 6. Ranking order is correct
    # ----------------------------------------------------------------
    print("\n6. Testing ranking order — ML job should rank highest...")

    runner5 = make_runner(min_score=0.0)
    result5 = run_with_mocks(runner5)

    print(f"\n   Ranked jobs:")
    for r in result5.ranked:
        scraped = next(
            (s for s in MOCK_SCRAPED if s.job_id == r.job_id), None
        )
        name = scraped.raw_title if scraped else r.job_id
        print(f"   #{r.rank} {name:<25} score={r.overall_score:.4f}")

    ml_result = next(
        (r for r in result5.ranked if r.job_id == "job001"), None
    )
    dv_result = next(
        (r for r in result5.ranked if r.job_id == "job004"), None
    )

    print()
    checks = [
        (result5.ranked[0].rank == 1,
            "First result has rank 1"),
        (all(result5.ranked[i].overall_score >=
             result5.ranked[i+1].overall_score
             for i in range(len(result5.ranked)-1)),
            "Results sorted descending by score"),
        (ml_result is not None and dv_result is not None and
         ml_result.overall_score > dv_result.overall_score,
            "ML Engineer scores higher than DevOps for ML resume"),
    ]
    for passed, label in checks:
        check(passed, label)

    runner5.close()

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
