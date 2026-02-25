# tests/test_integration_module5.py
"""
Module 5 Integration Test — full pipeline end-to-end.

Uses:
  - Real MatchingEngine    (Module 3)
  - Real RankingEngine     (Module 4)
  - Real JobStore          (:memory:)
  - Real TrackerApp        (headless via Textual Pilot)
  - Controlled ScrapedJob  (no network calls)
  - Real PipelineRunner    (wired to :memory: store)

Verifies:
  1. PipelineConfig builds correctly
  2. PipelineRunner.run() produces correct RunResult
  3. JobStore is populated with ranked, scored jobs
  4. Status updates persist through the full stack
  5. TrackerApp reflects live store state
  6. Filter views return correct subsets
  7. Full lifecycle: find → apply → interview → offer

Run with:
    python tests/test_integration_module5.py
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import asyncio
import numpy as np
from datetime    import datetime
from unittest.mock import patch

from src.modules.pipeline_config  import PipelineConfig
from src.modules.pipeline_runner  import PipelineRunner, RunResult
from src.modules.pipeline_scraper import ScrapedJob
from src.modules.job_store        import JobStore
from src.modules.tracker_ui       import TrackerApp
from src.modules.resume_parser    import ResumeProfile
from src.modules.job_parser       import JobProfile
from src.modules.matching_engine  import MatchingEngine
from src.modules.ranking_engine   import RankingEngine


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

EMBEDDING_DIM = 384

def _unit_vec(seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed % (2**31))
    v   = rng.randn(EMBEDDING_DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def make_resume() -> ResumeProfile:
    skills = ["python", "machine learning", "tensorflow", "sql", "pandas"]
    return ResumeProfile(
        version                = "1.0",
        hard_skills            = skills,
        soft_skills            = [],
        education              = "Bachelor's in CS",
        total_experience_years = 4.0,
        job_history            = ["ML Engineer (4 yrs)"],
        resume_embedding       = _unit_vec(1),
        skills_embeddings      = {s: _unit_vec(i+2) for i, s in enumerate(skills)},
        raw_text               = " ".join(skills),
        created_at             = datetime.now(),
    )


def make_scraped(job_id, title, required, seed=100) -> ScrapedJob:
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


MOCK_RESUME  = make_resume()
MOCK_SCRAPED = [
    make_scraped("job001", "ML Engineer",
                 ["python", "machine learning", "tensorflow"], seed=101),
    make_scraped("job002", "Data Scientist",
                 ["python", "sql", "pandas"],                  seed=102),
    make_scraped("job003", "Backend Engineer",
                 ["python", "docker", "postgres"],             seed=103),
    make_scraped("job004", "DevOps Engineer",
                 ["docker", "kubernetes", "terraform"],        seed=104),
]


def make_runner(min_score: float = 0.0) -> PipelineRunner:
    config = PipelineConfig(
        search_role = "ML Engineer",
        resume_path = "data/resumes/my_resume.pdf",
        min_score   = min_score,
        db_path     = ":memory:",
    )
    runner       = PipelineRunner(config)
    runner.store = JobStore(":memory:")
    return runner


def run_pipeline(runner: PipelineRunner) -> RunResult:
    with patch.object(runner, "_load_resume", return_value=MOCK_RESUME), \
         patch.object(runner, "_scrape_jobs", return_value=MOCK_SCRAPED):
        return runner.run()


def section(title: str):
    print(f"\n{title}")

def check(passed: bool, label: str):
    print(f"   {'✓' if passed else '⚠'} {label}")


# ------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("MODULE 5 INTEGRATION TEST — Full Pipeline End-to-End")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. PipelineConfig
    # ----------------------------------------------------------------
    section("1. PipelineConfig...")

    config = PipelineConfig(
        search_role    = "ML Engineer",
        resume_path    = "data/resumes/my_resume.pdf",
        min_score      = 0.40,
        top_n          = 10,
        job_sites      = ["linkedin", "indeed"],
        db_path        = ":memory:",
    )
    print(f"\n   {config.summary()}\n")

    checks = [
        (config.search_role  == "ML Engineer", "search_role correct"),
        (config.min_score    == 0.40,          "min_score correct"),
        (config.top_n        == 10,            "top_n correct"),
        (abs(sum(config.weights.values()) - 1.0) < 0.001,
                                               "weights sum to 1.0"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 2. PipelineRunner — full run
    # ----------------------------------------------------------------
    section("2. PipelineRunner full run...")

    runner = make_runner(min_score=0.0)
    result = run_pipeline(runner)

    print(f"\n   {result.summary()}")
    print(f"\n   Ranked results:")
    for r in result.ranked:
        scraped = next(
            (s for s in MOCK_SCRAPED if s.job_id == r.job_id), None
        )
        name = scraped.raw_title if scraped else r.job_id
        print(f"   #{r.rank}  {name:<28} score={r.overall_score:.4f}  "
              f"norm={r.normalized_score:.4f}  label={r.rank_label}")

    print()
    checks = [
        (result.error        is None,   "No error"),
        (result.jobs_found   == 4,      "jobs_found = 4"),
        (result.jobs_matched == 4,      "All 4 pass min_score=0.0"),
        (result.jobs_new     == 4,      "All 4 are new"),
        (result.jobs_saved   == 4,      "All 4 saved"),
        (result.top_match    != "—",    "top_match is set"),
        (result.run_duration  > 0,      "run_duration > 0"),
        (len(result.ranked)  == 4,      "ranked has 4 entries"),
        (result.ranked[0].rank == 1,    "First ranked has rank 1"),
        (all(result.ranked[i].overall_score >=
             result.ranked[i+1].overall_score
             for i in range(3)),        "Ranked descending by score"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 3. ML job ranks above DevOps for ML resume
    # ----------------------------------------------------------------
    section("3. Semantic ranking correctness...")

    ml_r  = next(r for r in result.ranked if r.job_id == "job001")
    dv_r  = next(r for r in result.ranked if r.job_id == "job004")

    print(f"\n   ML Engineer   score = {ml_r.overall_score:.4f}")
    print(f"   DevOps Eng    score = {dv_r.overall_score:.4f}\n")

    checks = [
        (ml_r.overall_score > dv_r.overall_score,
            "ML Engineer scores higher than DevOps"),
        (ml_r.rank < dv_r.rank,
            "ML Engineer ranked higher (lower rank number)"),
        (ml_r.normalized_score == 1.0,
            "Best job normalized to 1.0"),
        (dv_r.normalized_score == 0.0,
            "Worst job normalized to 0.0"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 4. JobStore state after run
    # ----------------------------------------------------------------
    section("4. JobStore state after run...")

    stats    = runner.store.get_stats()
    all_jobs = runner.store.get_all()

    print(f"\n   Total in store: {stats['total']}")
    print(f"   Not applied:    {stats['not_applied']}")
    print(f"   Best score:     {stats['best_score']}")
    print(f"   Mean score:     {stats['mean_score']}\n")

    checks = [
        (stats["total"]       == 4,   "4 jobs in store"),
        (stats["not_applied"] == 4,   "All default to NOT_APPLIED"),
        (stats["best_score"]  >  0.0, "best_score > 0"),
        (stats["mean_score"]  >  0.0, "mean_score > 0"),
        (all("matched_skills" in j
             and isinstance(j["matched_skills"], list)
             for j in all_jobs),      "All have matched_skills as list"),
        (all("url" in j
             for j in all_jobs),      "All have url"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 5. Full application lifecycle via JobStore
    # ----------------------------------------------------------------
    section("5. Full application lifecycle...")

    store = runner.store

    store.update_status("job001", "APPLIED")
    store.update_status("job001", "INTERVIEWING")
    store.update_status("job001", "OFFER_RECEIVED")
    store.update_status("job002", "APPLIED")
    store.update_status("job003", "APPLIED")
    store.update_status("job003", "REJECTED")
    store.add_note("job001", "Offer received — deadline Friday")

    stats2 = store.get_stats()
    job1   = store.get_job("job001")

    print(f"\n   After lifecycle updates:")
    print(f"   offer_received = {stats2['offer_received']}")
    print(f"   applied        = {stats2['applied']}")
    print(f"   rejected       = {stats2['rejected']}")
    print(f"   job001 status  = {job1['status']}")
    print(f"   job001 note    = {job1['notes']}\n")

    checks = [
        (stats2["offer_received"] == 1,                "1 offer"),
        (stats2["applied"]        == 1,                "1 still applied"),
        (stats2["rejected"]       == 1,                "1 rejected"),
        (job1["status"]           == "OFFER_RECEIVED", "job001 → OFFER_RECEIVED"),
        (job1["applied_date"]     is not None,         "applied_date set"),
        (job1["notes"]            == "Offer received — deadline Friday",
                                                       "note persisted"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 6. Filter views
    # ----------------------------------------------------------------
    section("6. Filter views...")

    active   = store.get_all(status_filter=["APPLIED", "INTERVIEWING"])
    offers   = store.get_all(status_filter=["OFFER_RECEIVED"])
    rejected = store.get_all(status_filter=["REJECTED"])
    all_j    = store.get_all()

    print(f"\n   All:                   {len(all_j)}")
    print(f"   Active (applied+int):  {len(active)}")
    print(f"   Offers:                {len(offers)}")
    print(f"   Rejected:              {len(rejected)}\n")

    checks = [
        (len(all_j)    == 4, "All = 4"),
        (len(active)   == 1, "Active = 1 (job002 applied)"),
        (len(offers)   == 1, "Offers = 1 (job001)"),
        (len(rejected) == 1, "Rejected = 1 (job003)"),
        (offers[0]["job_id"] == "job001", "Correct job in offers"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 7. TrackerApp reflects live store state
    # ----------------------------------------------------------------
    section("7. TrackerApp — headless UI integration...")

    async def run_ui_checks():
        config = PipelineConfig(
            search_role = "ML Engineer",
            resume_path = "data/resumes/my_resume.pdf",
            db_path     = ":memory:",
        )
        app = TrackerApp(store=store, config=config)

        async with app.run_test() as pilot:
            await pilot.pause(0.1)

            table      = app.query_one("DataTable")
            row_count  = table.row_count
            stats_text = app._last_stats_text
            filter_idx = app._filter_index

            # Cycle filter to "Offers" (index 3)
            for _ in range(3):
                await pilot.press("f")
                await pilot.pause(0.05)
            offers_idx      = app._filter_index
            offers_rows     = table.row_count

            return (
                row_count, stats_text,
                filter_idx, offers_idx, offers_rows,
            )

    (
        row_count, stats_text,
        filter_idx, offers_idx, offers_rows,
    ) = asyncio.run(run_ui_checks())

    print(f"\n   DataTable rows (all):    {row_count}")
    print(f"   Stats text snippet:      {stats_text[:60]}...")
    print(f"   Initial filter index:    {filter_idx}")
    print(f"   After 3x [f] (Offers):  {offers_idx}")
    print(f"   Rows in Offers filter:  {offers_rows}\n")

    checks = [
        (row_count   == 4,          "Table shows all 4 jobs"),
        ("ML Engineer" in stats_text, "Stats bar shows role"),
        (filter_idx  == 0,          "Starts at filter 0 (All)"),
        (offers_idx  == 3,          "3x [f] reaches Offers filter"),
        (offers_rows == 1,          "Offers filter shows 1 job"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 8. Duplicate run protection
    # ----------------------------------------------------------------
    section("8. Duplicate run protection...")

    runner2        = make_runner(min_score=0.0)
    runner2.store  = store   # use same store that already has 4 jobs
    result2        = run_pipeline(runner2)

    print(f"\n   Second run: new={result2.jobs_new}, "
          f"saved={result2.jobs_saved}\n")

    checks = [
        (result2.jobs_new  == 0, "Second run: 0 new jobs"),
        (store.count()     == 4, "Store still has exactly 4 jobs"),
    ]
    for passed, label in checks:
        check(passed, label)

    runner.close()
    runner2.close()

    print("\n" + "=" * 70)
    print("ALL INTEGRATION TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
