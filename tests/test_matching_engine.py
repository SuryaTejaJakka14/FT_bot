# tests/test_matching_engine.py
"""
Integration test for MatchingEngine.
Tests the full Module 3 pipeline end-to-end.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from datetime import datetime

from src.modules.matching_engine import MatchingEngine
from src.modules.embeddings_creator import EmbeddingsCreator
from src.modules.resume_parser import ResumeProfile
from src.modules.job_parser import JobProfile


def build_resume(creator, hard_skills, education, experience_years, job_history):
    """Helper: build a fully populated ResumeProfile."""
    summary = creator.build_resume_summary_text(
        hard_skills=hard_skills,
        soft_skills=[],
        education=education,
        experience_years=experience_years,
        job_history=job_history,
    )
    resume_embedding  = creator.create_text_embedding(summary)
    skills_embeddings = creator.create_skills_embeddings(hard_skills)

    return ResumeProfile(
        version="1.0",
        hard_skills=hard_skills,
        soft_skills=[],
        education=education,
        total_experience_years=experience_years,
        job_history=job_history,
        resume_embedding=resume_embedding,
        skills_embeddings=skills_embeddings,
        raw_text=summary,
        created_at=datetime.now(),
    )


def build_job(creator, title, required_skills, nice_to_have, experience_years, education):
    """Helper: build a fully populated JobProfile."""
    embeddings = creator.create_job_embeddings(
        title=title,
        required_hard_skills=required_skills,
        nice_to_have_skills=nice_to_have,
        required_experience_years=experience_years,
        required_education=education,
    )
    return JobProfile(
        version="1.0",
        title=title,
        company="Test Corp",
        location="Remote",
        required_hard_skills=required_skills,
        nice_to_have_skills=nice_to_have,
        required_experience_years=experience_years,
        required_education=education,
        job_embedding=embeddings["job_embedding"],
        skills_embeddings=embeddings["skills_embeddings"],
        created_at=datetime.now(),
    )


def main():
    print("=" * 70)
    print("TESTING MATCHING ENGINE — Full Pipeline")
    print("=" * 70)

    engine  = MatchingEngine()
    creator = EmbeddingsCreator()

    # Build resume: Senior ML Engineer
    ml_resume = build_resume(
        creator,
        hard_skills=["python", "tensorflow", "aws", "sql", "docker"],
        education="Master of Science in Computer Science",
        experience_years=5.0,
        job_history=["Senior ML Engineer at Google (2020-Present)"],
    )

    # ----------------------------------------------------------------
    # 1. Strong match: ML Engineer job
    # ----------------------------------------------------------------
    print("\n1. Testing strong match (ML resume vs ML Engineer job)...")

    ml_job = build_job(
        creator,
        title="Machine Learning Engineer",
        required_skills=["python", "tensorflow", "aws", "sql"],
        nice_to_have=["docker", "kubernetes"],
        experience_years=4.0,
        education="Master in Computer Science",
    )

    result = engine.match(ml_resume, ml_job)
    print(f"\n   Overall score:    {result.overall_score:.3f}")
    print(f"   Match label:      {result.get_match_label()}")
    print(f"   Semantic score:   {result.semantic_score:.3f}")
    print(f"   Skills score:     {result.skills_score:.3f}")
    print(f"   Experience score: {result.experience_score:.3f}")
    print(f"   Education score:  {result.education_score:.3f}")
    print(f"   Matched skills:   {result.matched_skills}")
    print(f"   Missing skills:   {result.missing_skills}")
    print(f"   Bonus skills:     {result.bonus_skills}")
    print(f"   Summary: {result.summary()}")

    checks = [
        (result.overall_score    >= 0.70, "Overall score >= 0.70"),
        (result.skills_score     >= 0.75, "Skills score >= 0.75"),
        (result.experience_score == 1.00, "Meets experience (5y >= 4y req)"),
        (result.education_score  == 1.00, "Meets education (Master >= Master)"),
        ("python" in result.matched_skills, "Python matched"),
        (result.missing_skills   == [],   "No missing required skills"),
        ("docker" in result.bonus_skills,  "Docker is bonus skill"),
    ]

    print("\n   Checks:")
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 2. Weak match: Frontend Developer job
    # ----------------------------------------------------------------
    print("\n2. Testing weak match (ML resume vs Frontend Developer job)...")

    frontend_job = build_job(
        creator,
        title="Frontend Developer",
        required_skills=["javascript", "react", "css", "typescript"],
        nice_to_have=["nodejs", "graphql"],
        experience_years=3.0,
        education="Bachelor in Computer Science",
    )

    result2 = engine.match(ml_resume, frontend_job)
    print(f"\n   Overall score:  {result2.overall_score:.3f}")
    print(f"   Match label:    {result2.get_match_label()}")
    print(f"   Skills score:   {result2.skills_score:.3f}")
    print(f"   Missing skills: {result2.missing_skills}")

    print(f"\n   {'✓' if result2.overall_score < 0.60 else '⚠'} Weak match score < 0.60")
    print(f"   {'✓' if result2.skills_score  < 0.30 else '⚠'} Low skills score for wrong domain")

    # ----------------------------------------------------------------
    # 3. No match: Accounting job
    # ----------------------------------------------------------------
    print("\n3. Testing no match (ML resume vs Accountant job)...")

    accounting_job = build_job(
        creator,
        title="Senior Accountant",
        required_skills=["excel", "quickbooks", "gaap", "accounting"],
        nice_to_have=["sql"],
        experience_years=3.0,
        education="Bachelor in Accounting or Finance",
    )

    result3 = engine.match(ml_resume, accounting_job)
    print(f"\n   Overall score: {result3.overall_score:.3f}")
    print(f"   Match label:   {result3.get_match_label()}")
    print(f"   Skills score:  {result3.skills_score:.3f}")
    print(f"\n   {'✓' if result3.overall_score < 0.50 else '⚠'} No match score < 0.50")

    # ----------------------------------------------------------------
    # 4. Score ranking validation
    # ----------------------------------------------------------------
    print("\n4. Validating score ranking...")

    scores = {
        "ML Engineer    (strong)": result.overall_score,
        "Frontend Dev   (weak)  ": result2.overall_score,
        "Accountant     (none)  ": result3.overall_score,
    }

    print(f"\n   {'Job':<35}  Score  Bar")
    print(f"   {'-'*35}  -----  ---")
    for title, score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 30)
        print(f"   {title:<35}  {score:.3f}  {bar}")

    ranking_correct = (
        result.overall_score > result2.overall_score > result3.overall_score
    )
    print(f"\n   {'✓' if ranking_correct else '⚠'} ML > Frontend > Accounting ranking correct")

    # ----------------------------------------------------------------
    # 5. Edge case: empty job profile
    # ----------------------------------------------------------------
    print("\n5. Testing edge case: empty job profile...")

    empty_job = JobProfile(
        version="1.0",
        title="",
        company="",
        location="",
        required_hard_skills=[],
        nice_to_have_skills=[],
        required_experience_years=0.0,
        required_education="",
        job_embedding=np.zeros(384),
        skills_embeddings={},
        created_at=datetime.now(),
    )

    result4 = engine.match(ml_resume, empty_job)
    print(f"   Empty job → overall_score: {result4.overall_score:.3f}")
    print(f"   {'✓' if result4.skills_score     == 1.0 else '⚠'} No required skills → skills_score = 1.0")
    print(f"   {'✓' if result4.experience_score == 1.0 else '⚠'} No experience req  → experience_score = 1.0")
    print(f"   {'✓' if result4.education_score  == 1.0 else '⚠'} No education req   → education_score = 1.0")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
