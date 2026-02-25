# tests/test_multi_resume_ranker.py
"""
Tests for MultiResumeRanker.
Uses a MockMatcher so the test is fully self-contained.
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.multi_resume_ranker import MultiResumeRanker


# ------------------------------------------------------------------
# Mock objects
# ------------------------------------------------------------------

class MockMatchResult:
    def __init__(self, overall_score: float):
        self.overall_score = overall_score


class MockMatcher:
    """Returns a pre-set score keyed on resume_id."""
    def __init__(self, score_map: dict):
        self.score_map = score_map

    def match(self, resume, job):
        resume_id = resume.get("resume_id", "unknown")
        score     = self.score_map.get(resume_id, 0.50)
        return MockMatchResult(overall_score=score)


def make_resumes(resume_ids):
    return [{"resume_id": rid} for rid in resume_ids]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

def main():
    print("=" * 70)
    print("TESTING MULTI-RESUME RANKER")
    print("=" * 70)

    score_map = {
        "alice":   0.91,
        "bob":     0.78,
        "charlie": 0.74,
        "diana":   0.61,
        "eve":     0.45,
    }
    matcher = MockMatcher(score_map)
    ranker  = MultiResumeRanker(matcher)
    job     = {"job_id": "ml_engineer"}
    resumes = make_resumes(["alice", "bob", "charlie", "diana", "eve"])

    # ----------------------------------------------------------------
    # 1. Core ranking
    # ----------------------------------------------------------------
    print("\n1. Testing core ranking...")

    results = ranker.rank(job, resumes)

    print(f"\n   {'Rank':<5} {'Candidate':<10} {'Score':>7}  {'Normalized':>11}  "
          f"{'Percentile':>11}  {'Label'}")
    print(f"   {'-'*5} {'-'*10} {'-'*7}  {'-'*11}  {'-'*11}  {'-'*12}")
    for r in results:
        print(f"   #{r.rank:<4} {r.resume_id:<10} {r.overall_score:>7.4f}  "
              f"{r.normalized_score:>11.4f}  {r.percentile:>11.4f}  {r.rank_label}")

    checks = [
        (len(results) == 5,                         "Returns 5 results"),
        (results[0].resume_id == "alice",            "Best candidate is alice"),
        (results[-1].resume_id == "eve",             "Worst candidate is eve"),
        (results[0].rank == 1,                       "Best candidate gets rank 1"),
        (results[-1].rank == 5,                      "Worst candidate gets rank 5"),
        (results[0].normalized_score == 1.0,         "Best → normalized 1.0"),
        (results[-1].normalized_score == 0.0,        "Worst → normalized 0.0"),
        (all(r.job_id == "" for r in results),       "job_id blank in multi-resume context"),
        (all(r.resume_id != "" for r in results),    "All results have resume_id"),
    ]
    print()
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 2. Sort order
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

    empty_result = ranker.rank(job, [])
    print(f"\n   Empty resumes → {empty_result}")
    print(f"   {'✓' if empty_result == [] else '⚠'} Empty resumes → []")

    single = ranker.rank(job, make_resumes(["alice"]))
    print(f"\n   Single resume:")
    print(f"     rank={single[0].rank}, normalized={single[0].normalized_score}, "
          f"percentile={single[0].percentile}")
    print(f"   {'✓' if single[0].rank == 1 else '⚠'} Single → rank 1")
    print(f"   {'✓' if single[0].normalized_score == 1.0 else '⚠'} Single → normalized 1.0")
    print(f"   {'✓' if single[0].percentile == 1.0 else '⚠'} Single → percentile 1.0")

    # ----------------------------------------------------------------
    # 4. rank_with_details()
    # ----------------------------------------------------------------
    print("\n4. Testing rank_with_details()...")

    details = ranker.rank_with_details(job, resumes)
    print(f"\n   Pool of 5 resumes:")
    for key in ["n", "best_candidate", "best_score", "worst_score",
                "score_range", "mean_score"]:
        print(f"   {key:<16}: {details[key]}")

    checks = [
        (details["n"]              == 5,       "n = 5"),
        (details["best_candidate"] == "alice", "best_candidate = alice"),
        (details["best_score"]     == 0.91,    "best_score = 0.91"),
        (details["worst_score"]    == 0.45,    "worst_score = 0.45"),
        (abs(details["score_range"] - 0.46) < 0.001, "score_range ≈ 0.46"),
    ]
    print()
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 5. get_shortlist()
    # ----------------------------------------------------------------
    print("\n5. Testing get_shortlist()...")

    # Top 3
    shortlist = ranker.get_shortlist(job, resumes, top_n=3)
    print(f"\n   Top 3: {[r.resume_id for r in shortlist]}")
    print(f"   {'✓' if len(shortlist) == 3 else '⚠'} Returns exactly 3")
    print(f"   {'✓' if shortlist[0].resume_id == 'alice' else '⚠'} Alice is first")

    # With min_score filter
    filtered = ranker.get_shortlist(job, resumes, top_n=10, min_score=0.75)
    print(f"\n   top_n=10, min_score=0.75: {[r.resume_id for r in filtered]}")
    print(f"   {'✓' if all(r.overall_score >= 0.75 for r in filtered) else '⚠'} "
      f"All scores >= 0.75")
    print(f"   {'✓' if len(filtered) == 2 else '⚠'} Only alice (0.91) and bob (0.78) qualify")

    # top_n smaller than qualifying pool
    small = ranker.get_shortlist(job, resumes, top_n=2, min_score=0.60)
    print(f"\n   top_n=2, min_score=0.60: {[r.resume_id for r in small]}")
    print(f"   {'✓' if len(small) == 2 else '⚠'} Returns at most top_n=2")

    # No qualifying candidates
    none_qualify = ranker.get_shortlist(job, resumes, top_n=5, min_score=0.99)
    print(f"\n   min_score=0.99 (no one qualifies): {none_qualify}")
    print(f"   {'✓' if none_qualify == [] else '⚠'} Returns [] when none qualify")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
