# tests/test_score_normalizer.py
"""
Tests for ScoreNormalizer.
Verifies min-max normalization, edge cases, and details output.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.score_normalizer import ScoreNormalizer


def main():
    print("=" * 70)
    print("TESTING SCORE NORMALIZER")
    print("=" * 70)

    normalizer = ScoreNormalizer()

    # ----------------------------------------------------------------
    # 1. Core normalization
    # ----------------------------------------------------------------
    print("\n1. Testing core min-max normalization...")

    raw    = [0.82, 0.78, 0.74, 0.71, 0.68]
    normed = normalizer.normalize(raw)

    print(f"\n   {'Raw':>7}  {'Normalized':>12}  Bar")
    print(f"   {'-'*7}  {'-'*12}  ---")
    for r, n in zip(raw, normed):
        bar = "█" * int(n * 25)
        print(f"   {r:>7.3f}  {n:>12.4f}  {bar}")

    checks = [
        (normed[0] == 1.0,                  "Best score → 1.0"),
        (normed[-1] == 0.0,                 "Worst score → 0.0"),
        (all(0.0 <= s <= 1.0 for s in normed), "All scores in [0, 1]"),
        (normed[0] > normed[1] > normed[2], "Order preserved"),
        (len(normed) == len(raw),           "Length preserved"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 2. Edge cases
    # ----------------------------------------------------------------
    print("\n2. Testing edge cases...")

    # All equal
    equal = normalizer.normalize([0.75, 0.75, 0.75])
    print(f"\n   All equal [0.75, 0.75, 0.75] → {equal}")
    print(f"   {'✓' if equal == [1.0, 1.0, 1.0] else '⚠'} All equal → [1.0, 1.0, 1.0]")

    # Single score
    single = normalizer.normalize([0.60])
    print(f"\n   Single score [0.60] → {single}")
    print(f"   {'✓' if single == [1.0] else '⚠'} Single score → [1.0]")

    # Empty list
    empty = normalizer.normalize([])
    print(f"\n   Empty [] → {empty}")
    print(f"   {'✓' if empty == [] else '⚠'} Empty → []")

    # Two scores
    two = normalizer.normalize([0.90, 0.60])
    print(f"\n   Two scores [0.90, 0.60] → {two}")
    print(f"   {'✓' if two == [1.0, 0.0] else '⚠'} Two scores → [1.0, 0.0]")

    # ----------------------------------------------------------------
    # 3. normalize_with_details()
    # ----------------------------------------------------------------
    print("\n3. Testing normalize_with_details()...")

    details = normalizer.normalize_with_details([0.82, 0.78, 0.74, 0.68])
    print(f"\n   Input: [0.82, 0.78, 0.74, 0.68]")
    for key, val in details.items():
        print(f"   {key:<15}: {val}")

    checks = [
        (details["normalized"][0]  == 1.0,  "Best → 1.0"),
        (details["normalized"][-1] == 0.0,  "Worst → 0.0"),
        (details["min_score"]      == 0.68, "min_score = 0.68"),
        (details["max_score"]      == 0.82, "max_score = 0.82"),
        (abs(details["score_range"] - 0.14) < 0.001, "score_range = 0.14"),
        (details["all_equal"]      == False,"all_equal = False"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 4. Context sensitivity demo
    # ----------------------------------------------------------------
    print("\n4. Demonstrating context sensitivity...")

    strong_pool = [0.90, 0.88, 0.85]
    weak_pool   = [0.45, 0.42, 0.40]

    strong_normed = normalizer.normalize(strong_pool)
    weak_normed   = normalizer.normalize(weak_pool)

    print(f"\n   Strong pool {strong_pool} → {strong_normed}")
    print(f"   Weak pool   {weak_pool}   → {weak_normed}")
    print(f"\n   Note: Both pools' best candidate normalizes to 1.0")
    print(f"   Raw score preserved in overall_score for absolute quality!")
    print(f"   ✓ Demonstrates why we keep both raw and normalized scores")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
