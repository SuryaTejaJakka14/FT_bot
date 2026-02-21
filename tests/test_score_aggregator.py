# tests/test_score_aggregator.py
"""
Tests for ScoreAggregator.
Verifies weighted aggregation, custom weights, and edge cases.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.score_aggregator import ScoreAggregator


def main():
    print("=" * 70)
    print("TESTING SCORE AGGREGATOR")
    print("=" * 70)

    aggregator = ScoreAggregator()

    # ----------------------------------------------------------------
    # 1. Core aggregation cases
    # ----------------------------------------------------------------
    print("\n1. Testing core aggregation cases...")
    print(f"\n   {'Semantic':>9} {'Skills':>7} {'Exp':>6} {'Edu':>6}  {'Score':>7}  {'Expected':>9}  Status")
    print(f"   {'-'*9} {'-'*7} {'-'*6} {'-'*6}  {'-'*7}  {'-'*9}  ------")

    cases = [
        # semantic, skills, exp, edu, expected, label
        (1.00, 1.00, 1.00, 1.00, 1.0000, "Perfect match"),
        (0.00, 0.00, 0.00, 0.00, 0.0000, "No match"),
        (0.82, 0.90, 1.00, 1.00, 0.9060, "Strong match"),    # 0.246+0.360+0.200+0.100
        (0.60, 0.50, 0.70, 1.00, 0.6200, "Moderate match"),  # 0.180+0.200+0.140+0.100
        (0.40, 0.20, 0.50, 0.30, 0.3300, "Weak match"),      # 0.120+0.080+0.100+0.030
        (1.00, 0.00, 1.00, 1.00, 0.6000, "No skills match"),
        (0.00, 1.00, 0.00, 0.00, 0.4000, "Skills only"),
    ]

    all_passed = True
    for sem, ski, exp, edu, expected, label in cases:
        score  = aggregator.aggregate(sem, ski, exp, edu)
        passed = abs(score - expected) < 0.001
        status = "✓" if passed else "⚠"
        print(f"   {sem:>9.2f} {ski:>7.2f} {exp:>6.2f} {edu:>6.2f}  {score:>7.4f}  {expected:>9.4f}  {status} {label}")
        if not passed:
            all_passed = False

    print(f"\n   {'✓ All cases passed!' if all_passed else '⚠ Some cases failed'}")

    # ----------------------------------------------------------------
    # 2. aggregate_with_details()
    # ----------------------------------------------------------------
    print("\n2. Testing aggregate_with_details()...")

    details = aggregator.aggregate_with_details(
        semantic_score=0.82,
        skills_score=0.90,
        experience_score=1.00,
        education_score=1.00,
    )

    print(f"\n   Input: semantic=0.82, skills=0.90, exp=1.00, edu=1.00")
    for key, val in details.items():
        if key != "weights":
            print(f"   {key:<22}: {val}")
    print(f"   {'weights':<22}: {details['weights']}")

    checks = [
        (abs(details["overall_score"]       - 0.9060) < 0.001, "overall_score = 0.906"),
        (abs(details["weighted_skills"]     - 0.3600) < 0.001, "weighted_skills = 0.360"),
        (abs(details["weighted_semantic"]   - 0.2460) < 0.001, "weighted_semantic = 0.246"),
        (abs(details["weighted_experience"] - 0.2000) < 0.001, "weighted_experience = 0.200"),
        (abs(details["weighted_education"]  - 0.1000) < 0.001, "weighted_education = 0.100"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 3. Custom weights
    # ----------------------------------------------------------------
    print("\n3. Testing custom weights...")

    skills_heavy = ScoreAggregator(weights={
        "semantic":   0.10,
        "skills":     0.70,
        "experience": 0.10,
        "education":  0.10,
    })

    edu_heavy = ScoreAggregator(weights={
        "semantic":   0.10,
        "skills":     0.10,
        "experience": 0.10,
        "education":  0.70,
    })

    # Candidate: great skills, poor education
    sem, ski, exp, edu = 0.70, 0.95, 0.80, 0.30

    default_score      = aggregator.aggregate(sem, ski, exp, edu)
    skills_heavy_score = skills_heavy.aggregate(sem, ski, exp, edu)
    edu_heavy_score    = edu_heavy.aggregate(sem, ski, exp, edu)

    print(f"\n   Scores: semantic={sem}, skills={ski}, exp={exp}, edu={edu}")
    print(f"   Default      (skills=0.40): {default_score:.4f}")
    print(f"   Skills-heavy (skills=0.70): {skills_heavy_score:.4f}")
    print(f"   Edu-heavy    (skills=0.10): {edu_heavy_score:.4f}")
    print(f"   {'✓' if skills_heavy_score > default_score > edu_heavy_score else '⚠'} "
          f"Weights affect score correctly")

    # ----------------------------------------------------------------
    # 4. Weight validation
    # ----------------------------------------------------------------
    print("\n4. Testing weight validation...")

    # Weights don't sum to 1.0
    try:
        ScoreAggregator(weights={
            "semantic": 0.50, "skills": 0.50,
            "experience": 0.50, "education": 0.50
        })
        print("   ⚠ Should have raised ValueError")
    except ValueError as e:
        print(f"   ✓ Invalid weights raise ValueError: '{str(e)[:50]}...'")

    # Missing key
    try:
        ScoreAggregator(weights={
            "semantic": 0.50, "skills": 0.50
        })
        print("   ⚠ Should have raised ValueError for missing keys")
    except ValueError as e:
        print(f"   ✓ Missing keys raise ValueError: '{str(e)[:50]}...'")

    # ----------------------------------------------------------------
    # 5. Edge cases
    # ----------------------------------------------------------------
    print("\n5. Testing edge cases...")

    score = aggregator.aggregate(1.5, 1.5, 1.5, 1.5)
    print(f"   Scores > 1.0 clamped → {score:.4f}")
    print(f"   {'✓' if score == 1.0 else '⚠'} Over-range clamped to 1.0")

    score = aggregator.aggregate(-0.5, -0.5, -0.5, -0.5)
    print(f"   Negative scores clamped → {score:.4f}")
    print(f"   {'✓' if score == 0.0 else '⚠'} Negative clamped to 0.0")

    score = aggregator.aggregate(None, None, None, None)
    print(f"   None inputs → {score:.4f}")
    print(f"   {'✓' if score == 0.0 else '⚠'} None inputs handled as 0.0")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
