# tests/test_job_store.py
"""
Tests for JobStore.
Uses a temporary in-memory SQLite database — no files written to disk.
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import json
from datetime import datetime
from unittest.mock import MagicMock

from src.modules.job_store import JobStore, APPLICATION_STATUSES


def check(passed: bool, label: str):
    print(f"   {'✓' if passed else '⚠'} {label}")


# ------------------------------------------------------------------
# MOCK HELPERS
# ------------------------------------------------------------------

def make_scraped_job(
    job_id    = "abc123def456789a",
    title     = "Senior ML Engineer",
    company   = "OpenAI",
    location  = "Remote",
    url       = "https://linkedin.com/jobs/123",
    source    = "linkedin",
    date_posted = "2026-02-22",
):
    job             = MagicMock()
    job.job_id      = job_id
    job.raw_title   = title
    job.company     = company
    job.location    = location
    job.url         = url
    job.source      = source
    job.date_posted = date_posted
    return job


def make_ranking_result(
    score         = 0.87,
    rank_label    = "Top 25%",
    matched       = None,
    missing       = None,
):
    result                            = MagicMock()
    result.overall_score              = score
    result.rank_label                 = rank_label
    result.match_result               = MagicMock()
    result.match_result.matched_skills = matched or ["python", "tensorflow"]
    result.match_result.missing_skills = missing or ["kubernetes"]
    return result


def make_store() -> JobStore:
    """In-memory SQLite store — isolated per test."""
    return JobStore(":memory:")


# ------------------------------------------------------------------
# TESTS
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TESTING JOB STORE")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Initialisation
    # ----------------------------------------------------------------
    print("\n1. Testing initialisation...")

    store = make_store()
    print(f"\n   Store created at: {store.db_path}")
    print(f"   Initial count:    {store.count()}\n")

    check(store.count() == 0, "Empty store has 0 jobs")
    check(store.get_all() == [], "get_all() returns [] on empty store")

    # ----------------------------------------------------------------
    # 2. save_job()
    # ----------------------------------------------------------------
    print("\n2. Testing save_job()...")

    job1    = make_scraped_job(job_id="job001", title="ML Engineer",   company="OpenAI")
    job2    = make_scraped_job(job_id="job002", title="Data Scientist", company="Google")
    result1 = make_ranking_result(score=0.87, rank_label="Top 25%")
    result2 = make_ranking_result(score=0.72, rank_label="Top 50%")

    inserted1 = store.save_job(job1, result1)
    inserted2 = store.save_job(job2, result2)
    duplicate = store.save_job(job1, result1)   # same job_id

    print(f"\n   inserted1 = {inserted1}")
    print(f"   inserted2 = {inserted2}")
    print(f"   duplicate = {duplicate}")
    print(f"   count     = {store.count()}\n")

    check(inserted1 == True,  "First insert → True")
    check(inserted2 == True,  "Second insert → True")
    check(duplicate == False, "Duplicate insert → False")
    check(store.count() == 2, "Store has 2 jobs after 3 inserts")

    # ----------------------------------------------------------------
    # 3. get_all() and field correctness
    # ----------------------------------------------------------------
    print("\n3. Testing get_all() field correctness...")

    jobs = store.get_all()
    top  = jobs[0]   # ordered by match_score DESC

    print(f"\n   Top job: {top['title']} @ {top['company']}")
    print(f"   score          = {top['match_score']}")
    print(f"   matched_skills = {top['matched_skills']}")
    print(f"   missing_skills = {top['missing_skills']}")
    print(f"   status         = {top['status']}")
    print(f"   status_label   = {top['status_label']}\n")

    check(top["title"]          == "ML Engineer",              "title correct")
    check(top["company"]        == "OpenAI",                   "company correct")
    check(top["match_score"]    == 0.87,                       "score correct")
    check(top["status"]         == "NOT_APPLIED",              "default status")
    check(top["status_label"]   == "○ Not Applied",            "status_label correct")
    check(isinstance(top["matched_skills"], list),             "matched_skills is list")
    check("python" in top["matched_skills"],                   "python in matched_skills")
    check(isinstance(top["missing_skills"], list),             "missing_skills is list")

    # ----------------------------------------------------------------
    # 4. update_status()
    # ----------------------------------------------------------------
    print("\n4. Testing update_status()...")

    ok = store.update_status("job001", "APPLIED")
    row = store.get_job("job001")

    print(f"\n   update_status returned: {ok}")
    print(f"   new status:             {row['status']}")
    print(f"   applied_date:           {row['applied_date']}")
    print(f"   status_label:           {row['status_label']}\n")

    check(ok == True,                        "update returns True")
    check(row["status"] == "APPLIED",        "status updated to APPLIED")
    check(row["applied_date"] is not None,   "applied_date set automatically")
    check(row["status_label"] == "✓ Applied","status_label correct")

    # Advance through lifecycle
    store.update_status("job001", "INTERVIEWING")
    check(
        store.get_job("job001")["status"] == "INTERVIEWING",
        "Status → INTERVIEWING"
    )
    store.update_status("job001", "OFFER_RECEIVED")
    check(
        store.get_job("job001")["status"] == "OFFER_RECEIVED",
        "Status → OFFER_RECEIVED"
    )

    # ----------------------------------------------------------------
    # 5. update_status() validation
    # ----------------------------------------------------------------
    print("\n5. Testing update_status() validation...")
    print()

    try:
        store.update_status("job001", "GHOST")
        check(False, "Should have raised ValueError for bad status")
    except ValueError:
        check(True, "Correctly rejects invalid status 'GHOST'")

    not_found = store.update_status("nonexistent_id", "APPLIED")
    check(not_found == False, "Returns False for unknown job_id")

    # ----------------------------------------------------------------
    # 6. get_all() with status_filter
    # ----------------------------------------------------------------
    print("\n6. Testing get_all() with status_filter...")

    store2 = make_store()
    for i, (status, score) in enumerate([
        ("NOT_APPLIED",  0.90),
        ("APPLIED",      0.80),
        ("INTERVIEWING", 0.75),
        ("REJECTED",     0.60),
        ("APPLIED",      0.70),
    ]):
        j = make_scraped_job(job_id=f"s2job{i:03d}", title=f"Job {i}")
        r = make_ranking_result(score=score)
        store2.save_job(j, r)
        if status != "NOT_APPLIED":
            store2.update_status(f"s2job{i:03d}", status)

    applied      = store2.get_all(status_filter=["APPLIED"])
    active       = store2.get_all(status_filter=["APPLIED", "INTERVIEWING"])
    all_jobs     = store2.get_all()
    high_scorers = store2.get_all(min_score=0.75)

    print(f"\n   APPLIED only:          {len(applied)}")
    print(f"   APPLIED+INTERVIEWING:  {len(active)}")
    print(f"   All:                   {len(all_jobs)}")
    print(f"   min_score >= 0.75:     {len(high_scorers)}\n")

    check(len(applied)      == 2, "2 APPLIED jobs")
    check(len(active)       == 3, "3 APPLIED+INTERVIEWING jobs")
    check(len(all_jobs)     == 5, "5 total jobs")
    check(len(high_scorers) == 3, "3 jobs with score >= 0.75")

    # ----------------------------------------------------------------
    # 7. add_note()
    # ----------------------------------------------------------------
    print("\n7. Testing add_note()...")

    store.add_note("job001", "Spoke to Sarah — follow up Friday")
    row = store.get_job("job001")
    print(f"\n   note = '{row['notes']}'\n")

    check(
        row["notes"] == "Spoke to Sarah — follow up Friday",
        "Note saved correctly"
    )
    check(
        store.add_note("nonexistent", "note") == False,
        "Returns False for unknown job_id"
    )

    # ----------------------------------------------------------------
    # 8. get_stats()
    # ----------------------------------------------------------------
    print("\n8. Testing get_stats()...")

    stats = store2.get_stats()
    print(f"\n   Stats: {stats}\n")

    check(stats["total"]        == 5,    "total = 5")
    check(stats["applied"]      == 2,    "applied = 2")
    check(stats["interviewing"] == 1,    "interviewing = 1")
    check(stats["rejected"]     == 1,    "rejected = 1")
    check(stats["not_applied"]  == 1,    "not_applied = 1")
    check(stats["best_score"]   == 0.90, "best_score = 0.90")
    check(stats["mean_score"]   >  0.0,  "mean_score > 0")

    # Empty store stats
    empty_stats = make_store().get_stats()
    check(empty_stats["total"] == 0, "Empty store stats: total = 0")

    # ----------------------------------------------------------------
    # 9. job_exists() + delete_job()
    # ----------------------------------------------------------------
    print("\n9. Testing job_exists() and delete_job()...")

    print(f"\n   job001 exists before delete: {store.job_exists('job001')}")
    store.delete_job("job001")
    print(f"   job001 exists after delete:  {store.job_exists('job001')}\n")

    check(store.job_exists("job002") == True,  "job002 exists")
    check(store.job_exists("job001") == False, "job001 gone after delete")
    check(
        store.delete_job("nonexistent") == False,
        "delete_job returns False for unknown id"
    )

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
