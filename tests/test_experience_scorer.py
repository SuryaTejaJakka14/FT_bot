# tests/test_experience_scorer.py
"""
Tests for ExperienceScorer.
Verifies penalty curve, edge cases, and score_with_details().
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.experience_scorer import ExperienceScorer


def main():
    print("=" * 70)
    print("TESTING EXPERIENCE SCORER")
    print("=" * 70)

    scorer = ExperienceScorer()

    # ----------------------------------------------------------------
    # 1. Core scoring cases
    # ----------------------------------------------------------------
    print("\n1. Testing core scoring cases...")
    print(f"\n   {'Candidate':>10}  {'Required':>10}  {'Score':>7}  {'Expected':>10}  Status")
    print(f"   {'-'*10}  {'-'*10}  {'-'*7}  {'-'*10}  ------")

    cases = [
        # (candidate, required, expected, label)
        (5.0, 5.0, 1.00, "Exact match"),
        (7.0, 5.0, 1.00, "Surplus"),
        (4.0, 5.0, 0.85, "1yr short"),
        (3.0, 5.0, 0.70, "2yrs short"),
        (2.0, 5.0, 0.55, "3yrs short"),
        (1.0, 5.0, 0.40, "4yrs short"),
        (0.0, 5.0, 0.25, "No experience"),
        (3.0, 0.0, 1.00, "No requirement"),
        (0.0, 0.0, 1.00, "Both zero"),
    ]

    all_passed = True
    for candidate, required, expected, label in cases:
        score  = scorer.score(candidate, required)
        passed = abs(score - expected) < 0.01
        status = "✓" if passed else "⚠"
        print(f"   {candidate:>10.1f}  {required:>10.1f}  {score:>7.3f}  {expected:>10.3f}  {status} {label}")
        if not passed:
            all_passed = False

    print(f"\n   {'✓ All cases passed!' if all_passed else '⚠ Some cases failed'}")

    # ----------------------------------------------------------------
    # 2. Penalty curve visualization
    # ----------------------------------------------------------------
    print("\n2. Penalty curve (required=5.0 years)...")
    print(f"\n   {'Candidate':>10}  {'Gap':>5}  {'Score':>7}  Bar")
    print(f"   {'-'*10}  {'-'*5}  {'-'*7}  ---")

    for candidate in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]:
        score = scorer.score(candidate, 5.0)
        gap   = max(0.0, 5.0 - candidate)
        bar   = "█" * int(score * 25)
        print(f"   {candidate:>10.1f}  {gap:>5.1f}  {score:>7.3f}  {bar}")

    # ----------------------------------------------------------------
    # 3. score_with_details()
    # ----------------------------------------------------------------
    print("\n3. Testing score_with_details()...")

    details = scorer.score_with_details(3.0, 5.0)
    print(f"\n   Candidate: 3.0 years | Required: 5.0 years")
    for key, val in details.items():
        print(f"   {key:<20}: {val}")

    checks = [
        (details["score"]             == 0.70,  "score = 0.70"),
        (details["gap_years"]         == 2.0,   "gap_years = 2.0"),
        (details["penalty_applied"]   == 0.30,  "penalty_applied = 0.30"),
        (details["meets_requirement"] == False, "meets_requirement = False"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # Meets requirement case
    details2 = scorer.score_with_details(6.0, 5.0)
    print(f"\n   Candidate: 6.0 years | Required: 5.0 years")
    print(f"   {'✓' if details2['meets_requirement'] else '⚠'} meets_requirement = True")
    print(f"   {'✓' if details2['gap_years'] == 0.0 else '⚠'} gap_years = 0.0")

    # ----------------------------------------------------------------
    # 4. Custom penalty rate
    # ----------------------------------------------------------------
    print("\n4. Testing custom penalty rate...")

    strict_scorer = ExperienceScorer(penalty_per_year=0.25)
    lenient_scorer = ExperienceScorer(penalty_per_year=0.05)

    candidate, required = 3.0, 5.0
    default_score = scorer.score(candidate, required)
    strict_score  = strict_scorer.score(candidate, required)
    lenient_score = lenient_scorer.score(candidate, required)

    print(f"\n   Candidate=3.0, Required=5.0, gap=2 years")
    print(f"   Lenient  (0.05/yr): {lenient_score:.3f}")
    print(f"   Default  (0.15/yr): {default_score:.3f}")
    print(f"   Strict   (0.25/yr): {strict_score:.3f}")

    print(f"   {'✓' if lenient_score > default_score > strict_score else '⚠'} Penalty rates work correctly")

    # ----------------------------------------------------------------
    # 5. Edge cases
    # ----------------------------------------------------------------
    print("\n5. Testing edge cases...")

    print(f"   None candidate → {scorer.score(None, 5.0):.3f}")
    print(f"   {'✓' if scorer.score(None, 5.0) == 0.25 else '⚠'} None treated as 0 years")

    print(f"   Negative years → {scorer.score(-2.0, 5.0):.3f}")
    print(f"   {'✓' if scorer.score(-2.0, 5.0) == 0.25 else '⚠'} Negative clamped to 0")

    print(f"   Very large gap → {scorer.score(0.0, 50.0):.3f}")
    print(f"   {'✓' if scorer.score(0.0, 50.0) == 0.0 else '⚠'} Floor at 0.0")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
