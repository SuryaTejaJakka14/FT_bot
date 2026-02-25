# tests/test_ranking_result.py
"""
Tests for RankingResult dataclass.
Verifies composition, delegation, rank labels, and convenience methods.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.ranking_result import RankingResult
from src.modules.match_result import MatchResult


def make_match_result(overall=0.85, skills=0.90, matched=None, missing=None, bonus=None):
    """Helper: build a MatchResult with given scores."""
    return MatchResult(
        overall_score=overall,
        semantic_score=0.80,
        skills_score=skills,
        experience_score=1.00,
        education_score=1.00,
        matched_skills=matched or ["python", "tensorflow", "aws"],
        missing_skills=missing or [],
        bonus_skills=bonus or ["docker"],
        skill_similarities={"python": 0.99, "tensorflow": 0.94},
    )


def main():
    print("=" * 70)
    print("TESTING RANKING RESULT DATACLASS")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Default values
    # ----------------------------------------------------------------
    print("\n1. Testing default values...")
    result = RankingResult()

    print(f"   rank:             {result.rank}")
    print(f"   resume_id:        '{result.resume_id}'")
    print(f"   job_id:           '{result.job_id}'")
    print(f"   match_result:     {result.match_result}")
    print(f"   percentile:       {result.percentile}")
    print(f"   normalized_score: {result.normalized_score}")
    print(f"   version:          {result.version}")
    print(f"   created_at:       {result.created_at}")

    checks = [
        (result.rank             == 0,    "rank defaults to 0"),
        (result.resume_id        == "",   "resume_id defaults to ''"),
        (result.percentile       == 0.0,  "percentile defaults to 0.0"),
        (result.normalized_score == 0.0,  "normalized_score defaults to 0.0"),
        (result.match_result     is None, "match_result defaults to None"),
        (result.version          == "1.0","version defaults to '1.0'"),
        (result.created_at       is not None, "created_at auto-populated"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 2. Composition — MatchResult embedded
    # ----------------------------------------------------------------
    print("\n2. Testing composition (MatchResult embedded)...")

    match = make_match_result(overall=0.950, skills=1.0)
    ranking = RankingResult(
        rank=1,
        resume_id="alice",
        job_id="ml_engineer",
        match_result=match,
        percentile=0.92,
        normalized_score=1.00,
    )

    print(f"   ranking.overall_score:   {ranking.overall_score}")
    print(f"   ranking.skills_score:    {ranking.skills_score}")
    print(f"   ranking.matched_skills:  {ranking.matched_skills}")
    print(f"   ranking.missing_skills:  {ranking.missing_skills}")
    print(f"   ranking.bonus_skills:    {ranking.bonus_skills}")

    checks = [
        (ranking.overall_score == 0.950,              "overall_score delegates to match_result"),
        (ranking.skills_score  == 1.0,                "skills_score delegates to match_result"),
        ("python" in ranking.matched_skills,           "matched_skills delegates to match_result"),
        (ranking.missing_skills == [],                 "missing_skills delegates to match_result"),
        ("docker" in ranking.bonus_skills,             "bonus_skills delegates to match_result"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 3. rank_label property
    # ----------------------------------------------------------------
    print("\n3. Testing rank_label property...")
    print(f"\n   {'Percentile':>12}  Label")
    print(f"   {'-'*12}  -----")

    label_cases = [
        (0.95, "Top 10%"),
        (0.90, "Top 10%"),
        (0.85, "Top 25%"),
        (0.75, "Top 25%"),
        (0.70, "Top 50%"),
        (0.50, "Top 50%"),
        (0.40, "Top 75%"),
        (0.25, "Top 75%"),
        (0.10, "Bottom 25%"),
        (0.00, "Bottom 25%"),
    ]

    all_passed = True
    for percentile, expected in label_cases:
        r = RankingResult(percentile=percentile)
        label = r.rank_label
        passed = label == expected
        print(f"   {percentile:>12.2f}  {label:<12}  {'✓' if passed else f'⚠ expected {expected}'}")
        if not passed:
            all_passed = False

    print(f"\n   {'✓ All label cases passed!' if all_passed else '⚠ Some label cases failed'}")

    # ----------------------------------------------------------------
    # 4. get_match_label() delegation
    # ----------------------------------------------------------------
    print("\n4. Testing get_match_label() delegation...")

    score_cases = [
        (0.85, "Excellent Match"),
        (0.70, "Good Match"),
        (0.55, "Partial Match"),
        (0.40, "Weak Match"),
        (0.20, "Poor Match"),
    ]

    for score, expected_label in score_cases:
        r = RankingResult(match_result=make_match_result(overall=score))
        label = r.get_match_label()
        status = "✓" if label == expected_label else "⚠"
        print(f"   {status} score={score} → '{label}'")

    # None match_result
    r_none = RankingResult()
    print(f"   {'✓' if r_none.get_match_label() == 'No Match Data' else '⚠'} None match_result → 'No Match Data'")
    print(f"   {'✓' if r_none.overall_score == 0.0 else '⚠'} None match_result → overall_score = 0.0")

    # ----------------------------------------------------------------
    # 5. summary()
    # ----------------------------------------------------------------
    print("\n5. Testing summary()...")

    r = RankingResult(
        rank=1,
        resume_id="alice",
        match_result=make_match_result(overall=0.950),
        percentile=0.92,
        normalized_score=1.00,
    )
    s = r.summary()
    print(f"   {s}")
    checks = [
        ("#1" in s,             "#1 rank in summary"),
        ("alice" in s,          "resume_id in summary"),
        ("0.950" in s,          "score in summary"),
        ("Top 10%" in s,        "rank_label in summary"),
        ("Excellent Match" in s,"match_label in summary"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 6. Mutable default isolation
    # ----------------------------------------------------------------
    print("\n6. Testing mutable default isolation...")
    r1 = RankingResult()
    r2 = RankingResult()
    # These use properties, no shared mutable state to test on RankingResult itself
    # But verify independent created_at timestamps
    isolated = r1.created_at is not None and r2.created_at is not None
    print(f"   {'✓' if isolated else '⚠'} created_at independently populated")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
