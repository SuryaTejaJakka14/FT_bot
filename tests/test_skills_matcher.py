# tests/test_skills_matcher.py
"""
Tests for SkillsMatcher.
Verifies exact matching, semantic matching, bonus skills,
and edge cases.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.skills_matcher import SkillsMatcher
from src.modules.embeddings_creator import EmbeddingsCreator


def main():
    print("=" * 70)
    print("TESTING SKILLS MATCHER")
    print("=" * 70)

    matcher = SkillsMatcher()
    creator = EmbeddingsCreator()

    # Pre-compute embeddings for all skills used in tests
    all_skills = [
        "python", "tensorflow", "pytorch", "aws", "sql",
        "docker", "kubernetes", "spark", "javascript",
        "react", "excel", "quickbooks", "machine learning",
    ]
    embeddings = creator.create_skills_embeddings(all_skills)

    # ----------------------------------------------------------------
    # 1. Exact match
    # ----------------------------------------------------------------
    print("\n1. Testing exact skill matching...")

    result = matcher.match(
        resume_skills=["python", "sql", "aws"],
        job_required_skills=["python", "sql", "aws"],
        job_nice_to_have_skills=[],
        resume_skill_embeddings=embeddings,
        job_skill_embeddings=embeddings,
    )

    print(f"   skills_score:    {result['skills_score']}")
    print(f"   matched_skills:  {result['matched_skills']}")
    print(f"   missing_skills:  {result['missing_skills']}")

    print(f"   {'✓' if result['skills_score'] == 1.0 else '⚠'} Perfect exact match = 1.0")
    print(f"   {'✓' if result['missing_skills'] == [] else '⚠'} No missing skills")

    # ----------------------------------------------------------------
    # 2. Semantic match (pytorch ≈ tensorflow)
    # ----------------------------------------------------------------
    print("\n2. Testing semantic matching (pytorch ≈ tensorflow)...")

    result = matcher.match(
        resume_skills=["python", "pytorch"],
        job_required_skills=["python", "tensorflow"],
        job_nice_to_have_skills=[],
        resume_skill_embeddings=embeddings,
        job_skill_embeddings=embeddings,
    )

    tf_score = result["skill_similarities"].get("tensorflow", 0)
    print(f"   pytorch ↔ tensorflow similarity: {tf_score:.3f}")
    print(f"   matched_skills: {result['matched_skills']}")
    print(f"   missing_skills: {result['missing_skills']}")
    print(f"   skills_score:   {result['skills_score']:.3f}")
    print(f"   {'✓' if tf_score >= 0.65 else '⚠'} pytorch ≈ tensorflow above threshold")

    # ----------------------------------------------------------------
    # 3. Partial match (some skills missing)
    # ----------------------------------------------------------------
    print("\n3. Testing partial match...")

    result = matcher.match(
        resume_skills=["python", "sql"],
        job_required_skills=["python", "sql", "kubernetes", "spark"],
        job_nice_to_have_skills=[],
        resume_skill_embeddings=embeddings,
        job_skill_embeddings=embeddings,
    )

    print(f"   skills_score:    {result['skills_score']:.3f}")
    print(f"   matched_skills:  {result['matched_skills']}")
    print(f"   missing_skills:  {result['missing_skills']}")

    print(f"   {'✓' if result['skills_score'] == 0.5 else '⚠'} 2/4 skills = 0.5")
    print(f"   {'✓' if 'python' in result['matched_skills'] else '⚠'} python matched")
    print(f"   {'✓' if 'kubernetes' in result['missing_skills'] else '⚠'} kubernetes missing")

    # ----------------------------------------------------------------
    # 4. Bonus skills (nice-to-have)
    # ----------------------------------------------------------------
    print("\n4. Testing bonus skills (nice-to-have)...")

    result = matcher.match(
        resume_skills=["python", "docker", "kubernetes"],
        job_required_skills=["python"],
        job_nice_to_have_skills=["docker", "kubernetes", "spark"],
        resume_skill_embeddings=embeddings,
        job_skill_embeddings=embeddings,
    )

    print(f"   bonus_skills:   {result['bonus_skills']}")
    print(f"   skills_score:   {result['skills_score']:.3f}")
    print(f"   {'✓' if 'docker' in result['bonus_skills'] else '⚠'} docker is a bonus skill")
    print(f"   {'✓' if 'kubernetes' in result['bonus_skills'] else '⚠'} kubernetes is a bonus skill")
    print(f"   {'✓' if result['skills_score'] == 1.0 else '⚠'} bonus skills dont affect main score")

    # ----------------------------------------------------------------
    # 5. No match (completely different domains)
    # ----------------------------------------------------------------
    print("\n5. Testing no match (ML resume vs Accounting job)...")

    result = matcher.match(
        resume_skills=["python", "tensorflow", "pytorch"],
        job_required_skills=["excel", "quickbooks"],
        job_nice_to_have_skills=[],
        resume_skill_embeddings=embeddings,
        job_skill_embeddings=embeddings,
    )

    print(f"   skills_score:   {result['skills_score']:.3f}")
    print(f"   matched_skills: {result['matched_skills']}")
    print(f"   missing_skills: {result['missing_skills']}")
    print(f"   similarities:   {result['skill_similarities']}")
    print(f"   {'✓' if result['skills_score'] == 0.0 else '⚠'} No match = 0.0")

    # ----------------------------------------------------------------
    # 6. Edge cases
    # ----------------------------------------------------------------
    print("\n6. Testing edge cases...")

    # No required skills
    r = matcher.match(
        resume_skills=["python"],
        job_required_skills=[],
        job_nice_to_have_skills=[],
        resume_skill_embeddings=embeddings,
        job_skill_embeddings=embeddings,
    )
    print(f"   No required skills → score: {r['skills_score']}")
    print(f"   {'✓' if r['skills_score'] == 1.0 else '⚠'} No requirements = 1.0")

    # Empty resume skills
    r = matcher.match(
        resume_skills=[],
        job_required_skills=["python", "sql"],
        job_nice_to_have_skills=[],
        resume_skill_embeddings={},
        job_skill_embeddings=embeddings,
    )
    print(f"   Empty resume → score: {r['skills_score']}, missing: {r['missing_skills']}")
    print(f"   {'✓' if r['skills_score'] == 0.0 else '⚠'} Empty resume = 0.0")
    print(f"   {'✓' if set(r['missing_skills']) == {'python', 'sql'} else '⚠'} All required skills missing")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
