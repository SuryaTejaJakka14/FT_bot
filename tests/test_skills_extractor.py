# tests/test_skills_extractor.py
"""
Test SkillsExtractor
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.skills_extractor import SkillsExtractor


def main():
    print("=" * 70)
    print("TESTING SKILLS EXTRACTOR")
    print("=" * 70)

    extractor = SkillsExtractor(use_nlp=True)
    print("\n1. Extractor initialized with NLP.")

    sample_text = """
    Senior Data Scientist with 5+ years of experience in Python, TensorFlow, and AWS.
    Built end-to-end machine learning pipelines and REST APIs using Flask and FastAPI.
    Deployed models to production on Kubernetes and Docker.
    Strong leadership, communication, and teamwork skills.
    Experienced with PostgreSQL, Git, and CI/CD.
    """

    print("\n2. Extracting skills from sample text...")
    hard_skills, soft_skills = extractor.extract_skills(sample_text)

    print(f"\n   Hard skills ({len(hard_skills)}):")
    for skill in hard_skills:
        print(f"     - {skill}")

    print(f"\n   Soft skills ({len(soft_skills)}):")
    for skill in soft_skills:
        print(f"     - {skill}")

    print("\n3. Sanity checks:")
    expected_hard = {
        "python", "tensorflow", "aws",
        "flask", "fastapi",
        "kubernetes", "docker",
        "postgresql", "git",
        "ci/cd", "machine learning", "rest api",
    }
    missing_hard = expected_hard - set(hard_skills)
    if missing_hard:
        print(f"   ⚠ Missing expected hard skills: {sorted(missing_hard)}")
    else:
        print("   ✓ All expected hard skills found (or normalized variants).")

    expected_soft = {"leadership", "communication", "teamwork"}
    missing_soft = expected_soft - set(soft_skills)
    if missing_soft:
        print(f"   ⚠ Missing expected soft skills: {sorted(missing_soft)}")
    else:
        print("   ✓ All expected soft skills found.")

    print("\n" + "=" * 70)
    print("TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
