# tests/test_percentile_calculator.py
"""
Tests for PercentileCalculator.
Verifies percentile formula, tie handling, and edge cases.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.percentile_calculator import PercentileCalculator


def main():
    print("=" * 70)
    print("TESTING PERCENTILE CALCULATOR")
    print("=" * 70)

    calc = PercentileCalculator()

    # ----------------------------------------------------------------
    # 1. Core percentile calculation
    # ----------------------------------------------------------------
    print("\n1. Testing core percentile calculation...")

    scores = [0.95, 0.82, 0.74, 0.61, 0.45]
    percs  = calc.calculate(scores)

    print(f"\n   {'Candidate':<10} {'Score':>7}  {'Percentile':>11}  {'Label':<12}")
    print(f"   {'-'*10} {'-'*7}  {'-'*11}  {'-'*12}")

    names = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
    expected_percs = [0.80, 0.60, 0.40, 0.20, 0.00]

    all_passed = True
    for name, score, perc, expected in zip(names, scores, percs, expected_percs):
        label  = calc.get_rank_label(perc)
        passed = abs(perc - expected) < 0.001
        status = "✓" if passed else "⚠"
        print(f"   {name:<10} {score:>7.3f}  {perc:>11.4f}  {label:<12}  {status}")
        if not passed:
            all_passed = False

    print(f"\n   {'✓ All cases passed!' if all_passed else '⚠ Some cases failed'}")

    # ----------------------------------------------------------------
    # 2. Tie handling
    # ----------------------------------------------------------------
    print("\n2. Testing tie handling...")

    tied_scores = [0.85, 0.82, 0.82, 0.70]
    tied_percs  = calc.calculate(tied_scores)

    print(f"\n   Scores: {tied_scores}")
    print(f"   Percentiles: {tied_percs}")

    checks = [
        (tied_percs[0] > tied_percs[1],       "0.85 > tied 0.82"),
        (tied_percs[1] == tied_percs[2],       "Both 0.82 get same percentile"),
        (tied_percs[3] == 0.0,                 "Lowest score → 0.0"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # All tied
    all_tied = calc.calculate([0.75, 0.75, 0.75])
    print(f"\n   All tied [0.75, 0.75, 0.75] → {all_tied}")
    print(f"   {'✓' if all(p == 0.0 for p in all_tied) else '⚠'} All tied → all 0.0 percentile")

    # ----------------------------------------------------------------
    # 3. Edge cases
    # ----------------------------------------------------------------
    print("\n3. Testing edge cases...")

    single = calc.calculate([0.80])
    print(f"\n   Single [0.80] → {single}")
    print(f"   {'✓' if single == [1.0] else '⚠'} Single → [1.0]")

    empty = calc.calculate([])
    print(f"\n   Empty [] → {empty}")
    print(f"   {'✓' if empty == [] else '⚠'} Empty → []")

    two = calc.calculate([0.90, 0.60])
    print(f"\n   Two [0.90, 0.60] → {two}")
    print(f"   {'✓' if two[0] > two[1] else '⚠'} Higher score → higher percentile")
    print(f"   {'✓' if two[1] == 0.0 else '⚠'} Lowest → 0.0")

    # ----------------------------------------------------------------
    # 4. calculate_with_details()
    # ----------------------------------------------------------------
    print("\n4. Testing calculate_with_details()...")

    details = calc.calculate_with_details([0.95, 0.82, 0.82, 0.70])
    print(f"\n   Input: [0.95, 0.82, 0.82, 0.70]")
    for key, val in details.items():
        print(f"   {key:<15}: {val}")

    checks = [
        (details["n"]            == 4,    "n = 4"),
        (details["top_score"]    == 0.95, "top_score = 0.95"),
        (details["bottom_score"] == 0.70, "bottom_score = 0.70"),
        (details["has_ties"]     == True, "has_ties = True"),
        (details["percentiles"][0] == 0.75, "0.95 → 75th percentile"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 5. get_rank_label()
    # ----------------------------------------------------------------
    print("\n5. Testing get_rank_label()...")
    print(f"\n   {'Percentile':>12}  Label")
    print(f"   {'-'*12}  -----")

    label_cases = [
        (0.95, "Top 10%"),
        (0.90, "Top 10%"),
        (0.80, "Top 25%"),
        (0.75, "Top 25%"),
        (0.60, "Top 50%"),
        (0.50, "Top 50%"),
        (0.30, "Top 75%"),
        (0.25, "Top 75%"),
        (0.10, "Bottom 25%"),
        (0.00, "Bottom 25%"),
    ]

    all_passed = True
    for perc, expected in label_cases:
        label  = calc.get_rank_label(perc)
        passed = label == expected
        print(f"   {perc:>12.2f}  {label:<12}  {'✓' if passed else f'⚠ expected {expected}'}")
        if not passed:
            all_passed = False

    print(f"\n   {'✓ All label cases passed!' if all_passed else '⚠ Some cases failed'}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
