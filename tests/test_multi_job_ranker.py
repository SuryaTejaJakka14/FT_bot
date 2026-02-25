# tests/test_multi_job_ranker.py
"""
Tests for MultiJobRanker.

Uses a MockMatcher so the test is fully self-contained
(no dependency on Module 3 during this isolated test).
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.multi_job_ranker import MultiJobRanker


# ------------------------------------------------------------------
# Mock objects
# ------------------------------------------------------------------

class MockMatchResult:
    """Minimal stand-in for a real MatchResult."""
    def __init__(self, overall_score: float, job_id: str = ""):
        self.overall_score = overall_score
        self.job_id        = job_id


class MockMatcher:
    """
    Returns a pre-set score for each job_id.
    Simulates what Module 3's Matcher would return.
    """
    def __init__(self, score_map: dict):
        self.score_map = score_map  # {job_id: score}

    def match(self, resume, job):
        job_id = job.get("job_id", "unknown")
        score  = self.score_map.get(job_id, 0.50)
        return MockMatchResult(overall_score=score, job_id=job_id)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def make_jobs(job_ids):
    return [{"job_id": jid} for jid in job_ids]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TESTING MULTI-JOB RANKER")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Core ranking
    # ----------------------------------------------------------------
    print("\n1. Testing core ranking...")

    score_map = {
        "job_001": 0.82,
        "job_002": 0.74,
        "job_003": 0.61,
        "job_004": 0.55,
        "job_005": 0.40,
    }
    matcher = MockMatcher(score_map)
    ranker  = MultiJobRanker(matcher)
    jobs    = make_jobs(["job_001", "job_002", "job_003", "job_004", "job_005"])

    results = ranker.rank("alice_resume", jobs)

    print(f"\n   {'Rank':<5} {'Job':<10} {'Score':>7}  {'Normalized':>11}  {'Percentile':>11}  {'Label'}")
    print(f"   {'-'*5} {'-'*10} {'-'*7}  {'-'*11}  {'-'*11}  {'-'*12}")
    for r in results:
        print(f"   #{r.rank:<4} {r.job_id:<10} {r.overall_score:>7.4f}  "
              f"{r.normalized_score:>11.4f}  {r.percentile:>11.4f}  {r.rank_label}")

    checks = [
        (len(results) == 5,                          "Returns 5 results"),
        (results[0].job_id == "job_001",             "Best job is job_001"),
        (results[-1].job_id == "job_005",            "Worst job is job_005"),
        (results[0].rank == 1,                       "Best job gets rank 1"),
        (results[-1].rank == 5,                      "Worst job gets rank 5"),
        (results[0].normalized_score == 1.0,         "Best → normalized 1.0"),
        (results[-1].normalized_score == 0.0,        "Worst → normalized 0.0"),
        (all(r.rank_label != "" for r in results),   "All results have rank labels"),
    ]
    print()
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 2. Sort order is correct
    # ----------------------------------------------------------------
    print("\n2. Testing sort order...")

    scores_in_order = [r.overall_score for r in results]
    is_descending   = all(
        scores_in_order[i] >= scores_in_order[i+1]
        for i in range(len(scores_in_order) - 1)
    )
    print(f"\n   Scores in result order: {scores_in_order}")
    print(f"   {'✓' if is_descending else '⚠'} Results are sorted descending by score")

    # ----------------------------------------------------------------
    # 3. Edge cases
    # ----------------------------------------------------------------
    print("\n3. Testing edge cases...")

    # Empty job list
    empty_result = ranker.rank("alice_resume", [])
    print(f"\n   Empty jobs → {empty_result}")
    print(f"   {'✓' if empty_result == [] else '⚠'} Empty jobs → []")

    # Single job
    single = ranker.rank("alice_resume", make_jobs(["job_001"]))
    print(f"\n   Single job:")
    print(f"     rank={single[0].rank}, normalized={single[0].normalized_score}, "
          f"percentile={single[0].percentile}")
    print(f"   {'✓' if single[0].rank == 1 else '⚠'} Single job → rank 1")
    print(f"   {'✓' if single[0].normalized_score == 1.0 else '⚠'} Single job → normalized 1.0")
    print(f"   {'✓' if single[0].percentile == 1.0 else '⚠'} Single job → percentile 1.0")

    # Two jobs
    two = ranker.rank("alice_resume", make_jobs(["job_001", "job_005"]))
    print(f"\n   Two jobs [job_001=0.82, job_005=0.40]:")
    for r in two:
        print(f"     #{r.rank} {r.job_id}: score={r.overall_score}, "
              f"norm={r.normalized_score}, pct={r.percentile}")
    print(f"   {'✓' if two[0].job_id == 'job_001' else '⚠'} Higher score ranks first")

    # ----------------------------------------------------------------
    # 4. rank_with_details()
    # ----------------------------------------------------------------
    print("\n4. Testing rank_with_details()...")

    details = ranker.rank_with_details("alice_resume", jobs)
    print(f"\n   Pool of 5 jobs:")
    for key in ["n", "best_job", "best_score", "worst_score", "score_range", "mean_score"]:
        print(f"   {key:<15}: {details[key]}")

    checks = [
        (details["n"]           == 5,        "n = 5"),
        (details["best_job"]    == "job_001", "best_job = job_001"),
        (details["best_score"]  == 0.82,      "best_score = 0.82"),
        (details["worst_score"] == 0.40,      "worst_score = 0.40"),
        (abs(details["score_range"] - 0.42) < 0.001, "score_range ≈ 0.42"),
    ]
    print()
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 5. match_result is preserved
    # ----------------------------------------------------------------
    print("\n5. Testing match_result passthrough...")

    result = ranker.rank("alice_resume", make_jobs(["job_001"]))[0]
    print(f"\n   result.match_result.overall_score = {result.match_result.overall_score}")
    print(f"   {'✓' if result.match_result.overall_score == 0.82 else '⚠'} "
          f"Raw MatchResult preserved inside RankingResult")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
