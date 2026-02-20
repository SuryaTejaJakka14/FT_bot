# tests/test_experience_extractor.py
"""
Test ExperienceExtractor
"""

import sys
import os
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.experience_extractor import ExperienceExtractor  # noqa: E402


def main():
    print("=" * 70)
    print("TESTING EXPERIENCE EXTRACTOR")
    print("=" * 70)

    # Fix current date so tests are stable
    fixed_now = datetime(2026, 1, 1)
    extractor = ExperienceExtractor(current_date=fixed_now)

    # ------------------------------------------------------------------
    # 1. Direct mention patterns
    # ------------------------------------------------------------------
    print("\n1. Testing direct mention patterns...")

    direct_cases = [
        ("I have 5 years of experience in software development.",
         5.0),
        ("Over 8 years of experience in data science.",
         8.0),
        ("More than 10 years working as an engineer.",
         10.0),
        ("3+ years experience in ML.",
         3.0),
        ("I bring 2.5 years of work experience.",
         2.5),
        ("No explicit years mentioned here.",
         0.0),
    ]

    for i, (text, expected) in enumerate(direct_cases, 1):
        years = extractor._extract_direct_mention(text)
        print(f"\n   Case {i}: '{text}'")
        print(f"   Expected ≥ {expected}, Got: {years}")
        if expected == 0.0:
            if years == 0.0:
                print("   ✓ Correctly returned 0.0")
            else:
                print("   ⚠ Should be 0.0")
        else:
            if abs(years - expected) < 0.1 or years >= expected:
                print("   ✓ Direct mention extracted correctly")
            else:
                print("   ⚠ Direct mention too low")

    # ------------------------------------------------------------------
    # 2. Date range extraction
    # ------------------------------------------------------------------
    print("\n2. Testing date range extraction...")

    date_text = """
    EXPERIENCE
    Software Engineer, Company A
    Jan 2018 - Dec 2020

    Senior Engineer, Company B
    Feb 2021 - Present

    Intern, Company C
    2015 - 2016
    """

    ranges = extractor.extract_all_date_ranges(date_text)
    print("\n   Extracted date ranges:")
    for r in ranges:
        print(f"     - {r['start']} to {r['end']} ({r['years']} years)")

    if len(ranges) >= 2:
        print("   ✓ Found multiple date ranges")
    else:
        print("   ⚠ Expected at least 2 ranges")

    # ------------------------------------------------------------------
    # 3. Full experience calculation (date ranges only)
    # ------------------------------------------------------------------
    print("\n3. Testing full experience calculation (date ranges)...")

    years_from_dates = extractor.extract_years_of_experience(date_text)
    print(f"   Years of experience (from dates): {years_from_dates:.1f}")

    # Rough expectations (with current_date=2026-01-01):
    # - Jan 2018 - Dec 2020 ~ 3.0 years
    # - Feb 2021 - Jan 2026 ~ 4.9 years
    # - 2015 - 2016 ~ 2.0 years (whole years)
    # We take max ≈ 4.9
    if 4.5 <= years_from_dates <= 5.5:
        print("   ✓ Years from date ranges in expected range (~5 years)")
    else:
        print("   ⚠ Years from date ranges look off")

    # ------------------------------------------------------------------
    # 4. Combined (direct mention + dates)
    # ------------------------------------------------------------------
    print("\n4. Testing combined extraction (direct + dates)...")

    combined_text = """
    I have over 6 years of experience in software engineering.
    EXPERIENCE
    Backend Engineer, X Corp
    Jan 2020 - Present
    """

    combined_years = extractor.extract_years_of_experience(combined_text)
    print(f"   Combined years: {combined_years:.1f}")

    # Direct mention: 6
    # Dates: Jan 2020 - Jan 2026 ~ 6.0 years
    # Max should be about 6
    if 5.5 <= combined_years <= 6.5:
        print("   ✓ Combined years in expected range (~6 years)")
    else:
        print("   ⚠ Combined years look off")

    # ------------------------------------------------------------------
    # 5. Edge cases
    # ------------------------------------------------------------------
    print("\n5. Testing edge cases...")

    edge_cases = [
        ("", 0.0),
        ("Experience: Present - Present", 0.0),
        ("Worked from 1990 - 2050 (nonsense future date)", 0.0),
        ("One year of experience", 0.0),  # no numeric
    ]

    for i, (text, expected) in enumerate(edge_cases, 1):
        years = extractor.extract_years_of_experience(text)
        print(f"\n   Edge case {i}: '{text}'")
        print(f"   Got: {years:.1f}, Expected: {expected:.1f}")
        if abs(years - expected) < 0.1:
            print("   ✓ Correct")
        else:
            print("   ⚠ Unexpected result")

    # ------------------------------------------------------------------
    # 6. Run on actual parsed resumes
    # ------------------------------------------------------------------
    print("\n6. Testing with actual parsed resume files...")

    from pathlib import Path

    parsed_dir = Path("data/resumes/parsed")
    if parsed_dir.exists():
        parsed_files = list(parsed_dir.glob("*_extracted.txt"))

        if parsed_files:
            print(f"   Found {len(parsed_files)} parsed resume files")

            for parsed_file in parsed_files:
                try:
                    with open(parsed_file, "r", encoding="utf-8") as f:
                        content = f.read()

                    years = extractor.extract_years_of_experience(content)
                    print(f"\n   File: {parsed_file.name}")
                    print(f"   Estimated years of experience: {years:.1f}")
                except Exception as e:
                    print(f"   ✗ Error reading {parsed_file.name}: {e}")
        else:
            print("   ⚠ No parsed resume files found in data/resumes/parsed")
    else:
        print(f"   ⚠ Parsed directory not found: {parsed_dir}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
