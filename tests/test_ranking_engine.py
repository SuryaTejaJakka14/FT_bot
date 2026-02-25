# tests/test_ranking_engine.py
"""
Tests for RankingEngine.
Verifies the facade correctly delegates to internal rankers.
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.ranking_engine import RankingEngine


# ------------------------------------------------------------------
# Mock objects
# ------------------------------------------------------------------

class MockMatchResult:
    def __init__(self, overall_score: float):
        self.overall_score = overall_score


class MockMatcher:
    def __init__(self, resume_scores: dict, job_scores: dict):
        # resume_scores: {resume_id: score}  used in rank_resumes_for_job
        # job_scores:    {job_id: score}     used in rank_jobs_for_resume
        self.resume_scores = resume_scores
        self.job_scores    = job_scores

    def match(self, resume, job):
        # Multi-resume mode: resume has resume_id
        if isinstance(resume, dict) and "resume_id" in resume:
            score = self.resume_scores.get(resume["resume_id"], 0.50)
        # Multi-job mode: job has job_id
        elif isinstance(job, dict) and "job_id" in job:
            score = self.job_scores.get(job["job_id"], 0.50)
        else:
            score = 0.50
        return MockMatchResult(overall_score=score)


def make_jobs(job_ids):
    return [{"job_id": jid} for jid in job_ids]

def make_resumes(resume_ids):
    return [{"resume_id": rid} for rid in resume_ids]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TESTING RANKING ENGINE")
    print("=" * 70)

    resume_scores = {
        "alice": 0.91, "bob": 0.78, "charlie": 0.74,
        "diana": 0.61, "eve": 0.45,
    }
    job_scores = {
        "job_ml":      0.88,
        "job_backend": 0.74,
        "job_data":    0.61,
        "job_devops":  0.52,
    }

    matcher = MockMatcher(resume_scores, job_scores)
    engine  = RankingEngine(matcher)

    resumes = make_resumes(["alice", "bob", "charlie", "diana", "eve"])
    jobs    = make_jobs(["job_ml", "job_backend", "job_data", "job_devops"])
    job     = {"job_id": "ml_engineer"}
    resume  = {"alice_resume"}

    # ----------------------------------------------------------------
    # 1. rank_jobs_for_resume()
    # ----------------------------------------------------------------
    print("\n1. Testing rank_jobs_for_resume()...")

    job_results = engine.rank_jobs_for_resume(resume, jobs)

    print(f"\n   {'Rank':<5} {'Job':<14} {'Score':>7}  {'Normalized':>11}  {'Label'}")
    print(f"   {'-'*5} {'-'*14} {'-'*7}  {'-'*11}  {'-'*12}")
    for r in job_results:
        print(f"   #{r.rank:<4} {r.job_id:<14} {r.overall_score:>7.4f}  "
              f"{r.normalized_score:>11.4f}  {r.rank_label}")

    checks = [
        (len(job_results) == 4,                    "Returns 4 job results"),
        (job_results[0].job_id == "job_ml",        "Best job is job_ml"),
        (job_results[0].rank == 1,                 "Best gets rank 1"),
        (job_results[0].normalized_score == 1.0,   "Best → normalized 1.0"),
        (job_results[-1].normalized_score == 0.0,  "Worst → normalized 0.0"),
    ]
    print()
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 2. rank_resumes_for_job()
    # ----------------------------------------------------------------
    print("\n2. Testing rank_resumes_for_job()...")

    resume_results = engine.rank_resumes_for_job(job, resumes)

    print(f"\n   {'Rank':<5} {'Candidate':<10} {'Score':>7}  {'Normalized':>11}  {'Label'}")
    print(f"   {'-'*5} {'-'*10} {'-'*7}  {'-'*11}  {'-'*12}")
    for r in resume_results:
        print(f"   #{r.rank:<4} {r.resume_id:<10} {r.overall_score:>7.4f}  "
              f"{r.normalized_score:>11.4f}  {r.rank_label}")

    checks = [
        (len(resume_results) == 5,                     "Returns 5 resume results"),
        (resume_results[0].resume_id == "alice",       "Best candidate is alice"),
        (resume_results[0].rank == 1,                  "Best gets rank 1"),
        (resume_results[0].normalized_score == 1.0,    "Best → normalized 1.0"),
        (resume_results[-1].normalized_score == 0.0,   "Worst → normalized 0.0"),
    ]
    print()
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 3. shortlist_resumes()
    # ----------------------------------------------------------------
    print("\n3. Testing shortlist_resumes()...")

    shortlist = engine.shortlist_resumes(job, resumes, top_n=3, min_score=0.75)
    print(f"\n   top_n=3, min_score=0.75: {[r.resume_id for r in shortlist]}")

    checks = [
        (len(shortlist) == 2,                              "Only 2 qualify (>=0.75)"),
        (shortlist[0].resume_id == "alice",                "Alice is first"),
        (all(r.overall_score >= 0.75 for r in shortlist),  "All scores >= 0.75"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 4. get_stats()
    # ----------------------------------------------------------------
    print("\n4. Testing get_stats()...")

    # Resumes mode
    stats = engine.get_stats("resumes", job, resumes)
    print(f"\n   mode='resumes':")
    for key in ["n", "best_candidate", "best_score", "worst_score", "mean_score"]:
        print(f"   {key:<16}: {stats[key]}")

    checks = [
        (stats["n"]              == 5,       "n = 5"),
        (stats["best_candidate"] == "alice", "best_candidate = alice"),
        (stats["best_score"]     == 0.91,    "best_score = 0.91"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # Jobs mode
    stats_jobs = engine.get_stats("jobs", resume, jobs)
    print(f"\n   mode='jobs':")
    for key in ["n", "best_job", "best_score", "worst_score", "mean_score"]:
        print(f"   {key:<16}: {stats_jobs[key]}")

    checks = [
        (stats_jobs["n"]        == 4,        "n = 4"),
        (stats_jobs["best_job"] == "job_ml", "best_job = job_ml"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # Invalid mode
    try:
        engine.get_stats("invalid", job, resumes)
        print("   ⚠ Should have raised ValueError")
    except ValueError:
        print("   ✓ Invalid mode raises ValueError")

    # ----------------------------------------------------------------
    # 5. Edge cases
    # ----------------------------------------------------------------
    print("\n5. Testing edge cases...")

    empty_jobs    = engine.rank_jobs_for_resume(resume, [])
    empty_resumes = engine.rank_resumes_for_job(job, [])
    empty_short   = engine.shortlist_resumes(job, [], top_n=5)

    print(f"\n   rank_jobs_for_resume(empty)    → {empty_jobs}")
    print(f"   rank_resumes_for_job(empty)    → {empty_resumes}")
    print(f"   shortlist_resumes(empty)       → {empty_short}")

    checks = [
        (empty_jobs    == [], "Empty jobs → []"),
        (empty_resumes == [], "Empty resumes → []"),
        (empty_short   == [], "Empty shortlist → []"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
