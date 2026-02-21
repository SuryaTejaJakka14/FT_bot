# tests/test_education_scorer.py
"""
Tests for EducationScorer.
Verifies level detection, scoring logic, and edge cases.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.education_scorer import EducationScorer


def main():
    print("=" * 70)
    print("TESTING EDUCATION SCORER")
    print("=" * 70)

    scorer = EducationScorer()

    # ----------------------------------------------------------------
    # 1. Level detection
    # ----------------------------------------------------------------
    print("\n1. Testing education level detection...")
    print(f"\n   {'Text':<45}  Level  Label")
    print(f"   {'-'*45}  -----  -----")

    detection_cases = [
        ("PhD in Machine Learning",                    5, "PhD"),
        ("Ph.D. in Computer Science",                  5, "PhD"),
        ("Master of Science in Data Science",          4, "Master"),
        ("MSc in Artificial Intelligence",             4, "Master"),
        ("Bachelor of Science in Computer Science",    3, "Bachelor"),
        ("BS in Mathematics",                          3, "Bachelor"),
        ("Associate degree in Information Technology", 2, "Associate"),
        ("High School Diploma",                        1, "High School"),
        ("No education info",                          0, "Unknown"),
        ("",                                           0, "Unknown"),
    ]

    for text, expected_level, expected_label in detection_cases:
        level = scorer.get_education_level(text)
        label = scorer._level_label(level)
        status = "✓" if level == expected_level else "⚠"
        display = text[:43] if len(text) > 43 else text
        print(f"   {display:<45}  {level:>5}  {label}  {status}")

    # ----------------------------------------------------------------
    # 2. Core scoring cases
    # ----------------------------------------------------------------
    print("\n2. Testing core scoring cases...")
    print(f"\n   {'Candidate':<12}  {'Required':<12}  {'Score':>7}  {'Expected':>9}  Status")
    print(f"   {'-'*12}  {'-'*12}  {'-'*7}  {'-'*9}  ------")

    score_cases = [
        # candidate edu,    required edu,    expected, label
        ("PhD",     "Bachelor", 1.00, "PhD > Bachelor"),
        ("Master",  "Bachelor", 1.00, "Master > Bachelor"),
        ("Bachelor","Bachelor", 1.00, "Exact match"),
        ("Bachelor","Master",   0.80, "1 level below"),
        ("Bachelor","PhD",      0.60, "2 levels below"),
        ("Associate","PhD",     0.40, "3 levels below"),
        ("",        "Bachelor", 0.30, "Unknown edu"),
        ("Master",  "",         1.00, "No requirement"),
        ("",        "",         1.00, "Both empty"),
    ]

    all_passed = True
    for cand, req, expected, label in score_cases:
        score  = scorer.score(cand, req)
        passed = abs(score - expected) < 0.01
        status = "✓" if passed else "⚠"
        print(f"   {cand:<12}  {req:<12}  {score:>7.3f}  {expected:>9.3f}  {status} {label}")
        if not passed:
            all_passed = False

    print(f"\n   {'✓ All cases passed!' if all_passed else '⚠ Some cases failed'}")

    # ----------------------------------------------------------------
    # 3. Real education strings
    # ----------------------------------------------------------------
    print("\n3. Testing real-world education strings...")

    real_cases = [
        (
            "Master of Science in Computer Science | Stanford University",
            "Bachelor's degree in Computer Science or related field",
            1.0,
            "MS > Bachelor"
        ),
        (
            "Bachelor of Engineering in Computer Science | MIT",
            "Master's degree in Computer Science or related field",
            0.80,
            "BS < Master"
        ),
        (
            "PhD in Machine Learning | Carnegie Mellon",
            "Master's degree preferred",
            1.0,
            "PhD > Master"
        ),
        (
            "Bachelor's degree in Computer Science, Mathematics or related field",
            "Bachelor's degree in Computer Science, Mathematics or related field",
            1.0,
            "Exact real-world strings"
        ),
    ]

    for cand, req, expected, label in real_cases:
        score = scorer.score(cand, req)
        status = "✓" if abs(score - expected) < 0.01 else "⚠"
        print(f"\n   {label}")
        print(f"   Candidate: '{cand[:55]}...'")
        print(f"   Required:  '{req[:55]}...'")
        print(f"   Score: {score:.3f}  {status}")

    # ----------------------------------------------------------------
    # 4. score_with_details()
    # ----------------------------------------------------------------
    print("\n4. Testing score_with_details()...")

    details = scorer.score_with_details(
        "Bachelor in Computer Science",
        "Master in Computer Science",
    )
    print(f"\n   Bachelor vs Master requirement:")
    for key, val in details.items():
        print(f"   {key:<20}: {val}")

    checks = [
        (details["score"]             == 0.80,    "score = 0.80"),
        (details["candidate_level"]   == 3,       "candidate_level = 3"),
        (details["required_level"]    == 4,       "required_level = 4"),
        (details["candidate_label"]   == "Bachelor", "candidate_label = Bachelor"),
        (details["required_label"]    == "Master",   "required_label = Master"),
        (details["gap_levels"]        == 1,       "gap_levels = 1"),
        (details["meets_requirement"] == False,   "meets_requirement = False"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
