# tests/test_match_result.py
"""
Tests for MatchResult dataclass.
Verifies fields, defaults, and convenience methods.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.match_result import MatchResult


def main():
    print("=" * 70)
    print("TESTING MATCH RESULT DATACLASS")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Default values
    # ----------------------------------------------------------------
    print("\n1. Testing default values...")
    result = MatchResult()

    print(f"   overall_score:    {result.overall_score}")
    print(f"   semantic_score:   {result.semantic_score}")
    print(f"   skills_score:     {result.skills_score}")
    print(f"   experience_score: {result.experience_score}")
    print(f"   education_score:  {result.education_score}")
    print(f"   matched_skills:   {result.matched_skills}")
    print(f"   missing_skills:   {result.missing_skills}")
    print(f"   bonus_skills:     {result.bonus_skills}")
    print(f"   version:          {result.version}")
    print(f"   created_at:       {result.created_at}")

    checks = [
        (result.overall_score    == 0.0,  "overall_score defaults to 0.0"),
        (result.matched_skills   == [],   "matched_skills defaults to []"),
        (result.skill_similarities == {}, "skill_similarities defaults to {}"),
        (result.version          == "1.0","version defaults to '1.0'"),
        (result.created_at       is not None, "created_at auto-populated"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 2. Mutable default isolation
    # ----------------------------------------------------------------
    print("\n2. Testing mutable default isolation...")
    r1 = MatchResult()
    r2 = MatchResult()
    r1.matched_skills.append("python")

    isolated = "python" not in r2.matched_skills
    print(f"   r1.matched_skills: {r1.matched_skills}")
    print(f"   r2.matched_skills: {r2.matched_skills}")
    print(f"   {'✓' if isolated else '⚠'} Lists are independent (no shared state)")

    # ----------------------------------------------------------------
    # 3. get_match_label()
    # ----------------------------------------------------------------
    print("\n3. Testing get_match_label()...")
    label_cases = [
        (0.85, "Excellent Match"),
        (0.70, "Good Match"),
        (0.55, "Partial Match"),
        (0.40, "Weak Match"),
        (0.20, "Poor Match"),
    ]
    for score, expected_label in label_cases:
        r = MatchResult(overall_score=score)
        label = r.get_match_label()
        status = "✓" if label == expected_label else "⚠"
        print(f"   {status} score={score} → '{label}' (expected '{expected_label}')")

    # ----------------------------------------------------------------
    # 4. is_strong_match()
    # ----------------------------------------------------------------
    print("\n4. Testing is_strong_match()...")
    cases = [
        (0.80, 0.75, True),
        (0.74, 0.75, False),
        (0.90, 0.90, True),
        (0.89, 0.90, False),
    ]
    for score, threshold, expected in cases:
        r = MatchResult(overall_score=score)
        result_val = r.is_strong_match(threshold)
        status = "✓" if result_val == expected else "⚠"
        print(f"   {status} score={score}, threshold={threshold} → {result_val}")

    # ----------------------------------------------------------------
    # 5. get_skills_coverage()
    # ----------------------------------------------------------------
    print("\n5. Testing get_skills_coverage()...")
    r = MatchResult(
        matched_skills=["python", "sql"],
        missing_skills=["kubernetes"],
    )
    coverage = r.get_skills_coverage()
    print(f"   matched=2, missing=1 → coverage={coverage:.3f}")
    print(f"   {'✓' if abs(coverage - 0.667) < 0.01 else '⚠'} Coverage correct (2/3 = 0.667)")

    empty_r = MatchResult()
    empty_coverage = empty_r.get_skills_coverage()
    print(f"   no skills → coverage={empty_coverage:.3f}")
    print(f"   {'✓' if empty_coverage == 1.0 else '⚠'} Empty skills → 1.0 (no requirements)")

    # ----------------------------------------------------------------
    # 6. summary()
    # ----------------------------------------------------------------
    print("\n6. Testing summary()...")
    r = MatchResult(
        overall_score=0.721,
        experience_score=0.80,
        education_score=1.00,
        matched_skills=["python", "tensorflow", "aws", "sql", "pytorch"],
        missing_skills=["kubernetes", "spark"],
    )
    summary = r.summary()
    print(f"   {summary}")
    print(f"   {'✓' if 'Good Match' in summary else '⚠'} Label in summary")
    print(f"   {'✓' if '0.721' in summary else '⚠'} Score in summary")
    print(f"   {'✓' if '5/7' in summary else '⚠'} Skills count in summary")

    # ----------------------------------------------------------------
    # 7. Full population test
    # ----------------------------------------------------------------
    print("\n7. Testing fully populated MatchResult...")
    full = MatchResult(
        overall_score=0.847,
        semantic_score=0.82,
        skills_score=0.90,
        experience_score=1.00,
        education_score=1.00,
        matched_skills=["python", "tensorflow", "aws"],
        missing_skills=["kubernetes"],
        bonus_skills=["docker"],
        skill_similarities={"python": 0.99, "tensorflow": 0.94, "aws": 0.88},
    )
    print(f"   overall_score:      {full.overall_score}")
    print(f"   match label:        {full.get_match_label()}")
    print(f"   skills coverage:    {full.get_skills_coverage():.3f}")
    print(f"   is_strong_match():  {full.is_strong_match()}")
    print(f"   summary: {full.summary()}")
    print(f"   {'✓' if full.get_match_label() == 'Excellent Match' else '⚠'} Excellent Match label")
    print(f"   {'✓' if full.is_strong_match() else '⚠'} is_strong_match() True")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
