# tests/test_job_embeddings.py

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.modules.embeddings_creator import EmbeddingsCreator


def main():
    print("=" * 70)
    print("TESTING JOB EMBEDDINGS BUILDER")
    print("=" * 70)

    creator = EmbeddingsCreator()

    # ----------------------------------------------------------------
    # 1. Test build_job_summary_text()
    # ----------------------------------------------------------------
    print("\n1. Testing build_job_summary_text()...")

    summary = creator.build_job_summary_text(
        title="Senior Data Scientist",
        required_hard_skills=["python", "tensorflow", "aws", "sql"],
        nice_to_have_skills=["docker", "kubernetes"],
        required_experience_years=5.0,
        required_education="Bachelor in Computer Science or related field",
    )

    print(f"   Summary:\n   '{summary}'\n")

    checks = [
        ("Senior Data Scientist" in summary, "Title included"),
        ("5.0"                   in summary, "Experience years included"),
        ("python"                in summary, "Required skill included"),
        ("docker"                in summary, "Nice-to-have skill included"),
        ("Bachelor"              in summary, "Education included"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 2. Test create_job_embeddings()
    # ----------------------------------------------------------------
    print("\n2. Testing create_job_embeddings()...")

    result = creator.create_job_embeddings(
        title="Senior Data Scientist",
        required_hard_skills=["python", "tensorflow", "aws", "sql"],
        nice_to_have_skills=["docker", "kubernetes"],
        required_experience_years=5.0,
        required_education="Bachelor in Computer Science or related field",
    )

    print(f"   job_embedding shape:  {result['job_embedding'].shape}")
    print(f"   skills embedded:      {list(result['skills_embeddings'].keys())}")
    print(f"   summary_text:         '{result['summary_text'][:60]}...'")

    shape_ok    = result["job_embedding"].shape == (384,)
    expected    = {"python", "tensorflow", "aws", "sql", "docker", "kubernetes"}
    skills_ok   = set(result["skills_embeddings"].keys()) == expected

    print(f"\n   {'✓' if shape_ok  else '⚠'} Embedding shape (384,)")
    print(f"   {'✓' if skills_ok else '⚠'} All 6 skills embedded (required + nice-to-have)")

    # ----------------------------------------------------------------
    # 3. Semantic similarity ranking
    # ----------------------------------------------------------------
    print("\n3. Testing resume ↔ job semantic similarity ranking...")

    resume_emb = creator.create_text_embedding(
        creator.build_resume_summary_text(
            hard_skills=["python", "tensorflow", "aws", "sql"],
            soft_skills=["leadership"],
            education="Master in Computer Science | Stanford",
            experience_years=5.0,
            job_history=["Senior Data Scientist at Google (2021-Present)"],
        )
    )

    jobs = {
        "ML Engineer       (strong match)": creator.create_text_embedding(
            creator.build_job_summary_text(
                title="Machine Learning Engineer",
                required_hard_skills=["python", "tensorflow", "pytorch", "aws"],
                nice_to_have_skills=["kubernetes", "docker"],
                required_experience_years=4.0,
                required_education="Master in Computer Science or related field",
            )
        ),
        "Data Scientist     (good match)": creator.create_text_embedding(
            creator.build_job_summary_text(
                title="Senior Data Scientist",
                required_hard_skills=["python", "tensorflow", "aws"],
                nice_to_have_skills=["docker"],
                required_experience_years=5.0,
                required_education="Master in Computer Science",
            )
        ),
        "Frontend Developer (weak match)": creator.create_text_embedding(
            creator.build_job_summary_text(
                title="Frontend Developer",
                required_hard_skills=["javascript", "react", "css"],
                nice_to_have_skills=["typescript"],
                required_experience_years=3.0,
                required_education="Bachelor in Computer Science",
            )
        ),
        "Accountant         (no match)  ": creator.create_text_embedding(
            creator.build_job_summary_text(
                title="Senior Accountant",
                required_hard_skills=["excel", "quickbooks", "gaap"],
                nice_to_have_skills=["sql"],
                required_experience_years=3.0,
                required_education="Bachelor in Accounting or Finance",
            )
        ),
    }

    print(f"\n   Resume: Senior Data Scientist | Python, TensorFlow, AWS\n")
    print(f"   {'Job':<40s} Score  Bar")
    print(f"   {'-'*40} -----  ---")

    scored = []
    for job_title, job_emb in jobs.items():
        score = creator.compute_similarity(resume_emb, job_emb)
        scored.append((job_title, score))

    for title, score in sorted(scored, key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 30)
        print(f"   {title:<40s} {score:.3f}  {bar}")

    # ----------------------------------------------------------------
    # 4. Edge cases
    # ----------------------------------------------------------------
    print("\n4. Testing edge cases...")

    empty = creator.build_job_summary_text(
        title="", required_hard_skills=[], nice_to_have_skills=[],
        required_experience_years=0.0, required_education=""
    )
    print(f"   Empty input  → '{empty}'")
    print(f"   {'✓' if 'No job information' in empty else '⚠'} Empty fallback text correct")

    no_pref = creator.build_job_summary_text(
        title="Backend Engineer", required_hard_skills=["python", "django"],
        nice_to_have_skills=[], required_experience_years=3.0, required_education=""
    )
    print(f"   No nice-to-have → '{no_pref}'")
    print(f"   {'✓' if 'Preferred' not in no_pref else '⚠'} Preferred section omitted when empty")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
