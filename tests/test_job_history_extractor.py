# tests/test_job_history_extractor.py
"""
Test JobHistoryExtractor
"""

import sys
import os
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.job_history_extractor import JobHistoryExtractor  # noqa: E402


def main():
    print("=" * 70)
    print("TESTING JOB HISTORY EXTRACTOR")
    print("=" * 70)

    fixed_now = datetime(2026, 1, 1)
    extractor = JobHistoryExtractor(use_nlp=True, current_date=fixed_now)

    # ------------------------------------------------------------------
    # 1. Simple multi-job resume text
    # ------------------------------------------------------------------
    print("\n1. Testing basic multi-job extraction...")

    sample_text = """
    JOHN DOE
    Senior Data Scientist

    EXPERIENCE

    Senior Data Scientist, Google
    Feb 2021 - Present

    Data Scientist | Microsoft
    Jan 2018 - Dec 2020

    Software Engineer
    Amazon
    2015 - 2017
    """

    entries = extractor.extract_job_history(sample_text)

    print(f"   Found {len(entries)} job entries")
    for e in entries:
        print(f"   - {e.format()}")

    if len(entries) >= 3:
        print("   ✓ Detected multiple job entries")
    else:
        print("   ⚠ Expected at least 3 job entries")

    # ------------------------------------------------------------------
    # 2. Check ordering (most recent first)
    # ------------------------------------------------------------------
    print("\n2. Checking ordering by start date (most recent first)...")

    if entries and len(entries) >= 2:
        first = entries[0]
        second = entries[1]
        if (first.start_date and second.start_date and
                first.start_date >= second.start_date):
            print("   ✓ Entries ordered from most recent to older")
        else:
            print("   ⚠ Entries may not be correctly ordered")

    # ------------------------------------------------------------------
    # 3. Edge cases
    # ------------------------------------------------------------------
    print("\n3. Testing edge cases...")

    edge_cases = [
        ("", 0),
        ("No experience section here", 0),
        ("Intern, Small Startup\n2024 - Present", 1),
    ]

    for i, (text, expected_min) in enumerate(edge_cases, 1):
        e_list = extractor.extract_job_history(text)
        print(f"\n   Edge case {i}: '{text[:40]}...'")
        print(f"   Found {len(e_list)} entries (expected ≥ {expected_min})")
        for e in e_list:
            print(f"   - {e.format()}")

        if len(e_list) >= expected_min:
            print("   ✓ OK")
        else:
            print("   ⚠ Fewer entries than expected")

    # ------------------------------------------------------------------
    # 4. Run on actual parsed resumes
    # ------------------------------------------------------------------
    print("\n4. Testing with actual parsed resume files...")

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

                    print(f"\n   File: {parsed_file.name}")
                    e_list = extractor.extract_job_history(content)
                    if not e_list:
                        print("   (No job entries found)")
                    else:
                        for e in e_list:
                            print(f"   - {e.format()}")
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
