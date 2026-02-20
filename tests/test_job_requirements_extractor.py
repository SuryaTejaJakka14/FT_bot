# tests/test_job_requirements_extractor.py
"""
Test JobRequirementsExtractor
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.job_requirements_extractor import JobRequirementsExtractor  # noqa: E402


SAMPLE_JD = """
Senior Data Scientist
Example Corp | Remote, USA

About the Role
We are looking for a Senior Data Scientist to join our growing AI team.
You will design and implement machine learning models at scale.

Responsibilities
- Build and deploy ML models in production
- Collaborate with cross-functional teams
- Design A/B tests and analyze results

Requirements
- 5+ years of experience in data science or machine learning
- Proficiency in Python, TensorFlow, and PyTorch
- Strong knowledge of SQL and data pipelines
- Experience with AWS or GCP cloud platforms
- Bachelor's degree in Computer Science, Mathematics or related field

Nice to Have
- Experience with Docker and Kubernetes
- Familiarity with Apache Spark
- Master's degree or PhD preferred
"""


def main():
    print("=" * 70)
    print("TESTING JOB REQUIREMENTS EXTRACTOR")
    print("=" * 70)

    extractor = JobRequirementsExtractor()

    # 1. Basic extraction
    print("\n1. Testing extraction from structured JD...")
    result = extractor.extract(SAMPLE_JD)

    print(f"\n   Title:              {result['title']}")
    print(f"   Company:            {result['company']}")
    print(f"   Location:           {result['location']}")
    print(f"   Required skills:    {result['required_hard_skills']}")
    print(f"   Nice-to-have:       {result['nice_to_have_skills']}")
    print(f"   Req experience:     {result['required_experience_years']} years")
    print(f"   Req education:      {result['required_education']}")

    # Validations
    print("\n   Checks:")

    if result["title"]:
        print(f"   ✓ Title extracted: '{result['title']}'")
    else:
        print("   ⚠ No title extracted")

    if result["required_hard_skills"]:
        print(f"   ✓ Required skills found: {len(result['required_hard_skills'])}")
    else:
        print("   ⚠ No required skills found")

    if result["required_experience_years"] > 0:
        print(f"   ✓ Experience years found: {result['required_experience_years']}")
    else:
        print("   ⚠ No experience years found")

    if result["required_education"]:
        print(f"   ✓ Education found: '{result['required_education'][:60]}...'")
    else:
        print("   ⚠ No education requirement found")

    # 2. Section splitting
    print("\n2. Testing section splitting...")

    sections = extractor._split_into_sections(SAMPLE_JD)
    print(f"   Sections found: {[k for k, v in sections.items() if v.strip()]}")

    if sections["required"].strip():
        print("   ✓ Required section extracted")
    else:
        print("   ⚠ Required section empty")

    if sections["preferred"].strip():
        print("   ✓ Preferred section extracted")
    else:
        print("   ⚠ Preferred section empty")

    # 3. Experience patterns
    print("\n3. Testing experience year patterns...")

    exp_cases = [
        ("5+ years of experience in data science", 5.0),
        ("Minimum 3 years of relevant experience", 3.0),
        ("2-4 years of experience", 2.0),
        ("At least 7 years working with Python", 7.0),
        ("No experience requirement mentioned", 0.0),
    ]

    for text, expected in exp_cases:
        got = extractor._extract_experience_years(text)
        status = "✓" if got == expected else "⚠"
        print(f"   {status} '{text[:45]}...' → {got} (expected {expected})")

    # 4. Education extraction
    print("\n4. Testing education extraction...")

    edu_cases = [
        "Bachelor's degree in Computer Science or related field",
        "BS/MS in Computer Science, Mathematics, or similar",
        "PhD in Machine Learning preferred",
        "This line has no education info",
    ]

    for text in edu_cases:
        result_edu = extractor._extract_education_requirement(text)
        print(f"   Input:  '{text}'")
        print(f"   Result: '{result_edu}'")
        print()

    # 5. Edge cases
    print("\n5. Testing edge cases...")

    empty_result = extractor.extract("")
    print(f"   Empty input → required skills: {empty_result['required_hard_skills']}")
    if not empty_result["required_hard_skills"]:
        print("   ✓ Empty input handled correctly")

    no_sections = extractor.extract(
        "We need a Python and SQL expert with 3 years of experience."
    )
    print(f"\n   No-section JD → required skills: {no_sections['required_hard_skills']}")
    print(f"   No-section JD → experience: {no_sections['required_experience_years']}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
