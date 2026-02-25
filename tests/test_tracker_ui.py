# tests/test_tracker_ui.py
"""
Tests for TrackerApp.

Uses Textual's Pilot API for headless testing — no real terminal needed.
Tests verify table population, keyboard shortcuts, status updates,
filter cycling, and store integration.
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import asyncio
from unittest.mock import MagicMock

from src.modules.job_store       import JobStore, STATUS_LABELS
from src.modules.pipeline_config import PipelineConfig
from src.modules.tracker_ui      import TrackerApp, FILTER_CYCLE


# ------------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------------

def check(passed: bool, label: str):
    print(f"   {'✓' if passed else '⚠'} {label}")


def run_async(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def make_config() -> PipelineConfig:
    return PipelineConfig(
        search_role = "ML Engineer",
        resume_path = "data/resumes/my_resume.pdf",
        db_path     = ":memory:",
    )


def make_store_with_jobs() -> JobStore:
    """In-memory store pre-loaded with 5 test jobs."""
    store = JobStore(":memory:")

    for i, (title, company, score, status) in enumerate([
        ("Senior ML Engineer",   "OpenAI",  0.91, "NOT_APPLIED"),
        ("ML Research Engineer", "Google",  0.85, "NOT_APPLIED"),
        ("Applied Scientist",    "Amazon",  0.78, "APPLIED"),
        ("Data Scientist",       "Meta",    0.71, "INTERVIEWING"),
        ("ML Platform Engineer", "Stripe",  0.65, "REJECTED"),
    ]):
        job             = MagicMock()
        job.job_id      = f"job{i:03d}"
        job.raw_title   = title
        job.company     = company
        job.location    = "Remote"
        job.url         = f"https://example.com/job{i}"
        job.source      = "linkedin"
        job.date_posted = "2026-02-24"

        result                             = MagicMock()
        result.overall_score               = score
        result.rank_label                  = "Top 25%"
        result.match_result                = MagicMock()
        result.match_result.matched_skills = ["python"]
        result.match_result.missing_skills = []

        store.save_job(job, result)
        if status != "NOT_APPLIED":
            store.update_status(f"job{i:03d}", status)

    return store


# ------------------------------------------------------------------
# ASYNC TEST FUNCTIONS
# ------------------------------------------------------------------

async def test_table_population():
    store  = make_store_with_jobs()
    config = make_config()
    app    = TrackerApp(store=store, config=config)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        table = app.query_one("DataTable")
        return table.row_count


async def test_key_apply():
    store  = make_store_with_jobs()
    config = make_config()
    app    = TrackerApp(store=store, config=config)

    async with app.run_test() as pilot:
        await pilot.press("home")
        await pilot.press("a")
        await pilot.pause(0.1)

    return store.get_job("job000")["status"]


async def test_key_interview():
    store  = make_store_with_jobs()
    config = make_config()
    app    = TrackerApp(store=store, config=config)

    async with app.run_test() as pilot:
        await pilot.press("home")
        await pilot.press("i")
        await pilot.pause(0.1)

    return store.get_job("job000")["status"]


async def test_filter_cycle():
    store  = make_store_with_jobs()
    config = make_config()
    app    = TrackerApp(store=store, config=config)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        initial   = app._filter_index
        await pilot.press("f")
        await pilot.pause(0.1)
        after_one = app._filter_index
        for _ in range(len(FILTER_CYCLE) - 1):
            await pilot.press("f")
            await pilot.pause(0.05)
        final = app._filter_index

    return initial, after_one, final


async def test_stats_bar():
    store  = make_store_with_jobs()
    config = make_config()
    app    = TrackerApp(store=store, config=config)

    async with app.run_test() as pilot:
        await pilot.pause(0.1)
        return app._last_stats_text    # plain text stored by _update_stats_bar


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TESTING TRACKER UI")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Table population
    # ----------------------------------------------------------------
    print("\n1. Testing table population...")

    row_count = run_async(test_table_population())
    print(f"\n   DataTable row count: {row_count}\n")
    check(row_count == 5, "Table has 5 rows (one per job)")

    # ----------------------------------------------------------------
    # 2. Key [a] → APPLIED
    # ----------------------------------------------------------------
    print("\n2. Testing [a] → APPLIED...")

    status = run_async(test_key_apply())
    print(f"\n   job000 status after [a]: {status}\n")
    check(status == "APPLIED", "[a] sets status to APPLIED")

    # ----------------------------------------------------------------
    # 3. Key [i] → INTERVIEWING
    # ----------------------------------------------------------------
    print("\n3. Testing [i] → INTERVIEWING...")

    status = run_async(test_key_interview())
    print(f"\n   job000 status after [i]: {status}\n")
    check(status == "INTERVIEWING", "[i] sets status to INTERVIEWING")

    # ----------------------------------------------------------------
    # 4. Filter cycle
    # ----------------------------------------------------------------
    print("\n4. Testing [f] filter cycle...")

    initial, after_one, final = run_async(test_filter_cycle())
    print(f"\n   initial filter index:  {initial}")
    print(f"   after one [f] press:   {after_one}")
    print(f"   after full cycle:      {final}\n")

    checks = [
        (initial   == 0, "Starts at filter index 0 (All)"),
        (after_one == 1, "One [f] press → index 1"),
        (final     == 0, "Full cycle returns to index 0"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 5. Stats bar content
    # ----------------------------------------------------------------
    print("\n5. Testing stats bar content...")

    stats_text = run_async(test_stats_bar())
    print(f"\n   Stats bar text: {stats_text}\n")

    checks = [
        ("ML Engineer" in stats_text, "Shows search role"),
        ("Total"       in stats_text, "Shows total count"),
        ("Applied"     in stats_text, "Shows applied count"),
        ("Filter"      in stats_text, "Shows filter name"),
        ("Showing"     in stats_text, "Shows row count"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 6. Status labels in store rows
    # ----------------------------------------------------------------
    print("\n6. Testing status labels in store rows...")

    store = make_store_with_jobs()
    jobs  = store.get_all()

    print(f"\n   All jobs with status labels:")
    for j in jobs:
        print(
            f"   {j['title']:<28} "
            f"{j['status']:<16} → {j['status_label']}"
        )

    print()
    checks = [
        (all("status_label" in j for j in jobs),
            "All rows have status_label"),
        (any(j["status_label"] == "✓ Applied"
             for j in jobs),
            "APPLIED label correct"),
        (any(j["status_label"] == "⟳ Interviewing"
             for j in jobs),
            "INTERVIEWING label correct"),
        (any(j["status_label"] == "✗ Rejected"
             for j in jobs),
            "REJECTED label correct"),
    ]
    for passed, label in checks:
        check(passed, label)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
