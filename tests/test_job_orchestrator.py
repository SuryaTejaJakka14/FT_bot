# tests/test_job_orchestrator.py
"""
Integration test for JobOrchestrator.
Verifies the full Module 2 pipeline end-to-end.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.job_orchestrator import JobOrchestrator


SAMPLE_JD = """
Senior Data Scientist
Example Corp | Remote, USA

About the Role:
We are looking for a Senior Data Scientist to join our AI team.

Requirements:
- 5+ years of experience in data science or machine learning
- Strong proficiency in Python, SQL, and TensorFlow
- Experience with AWS and GCP cloud platforms
- Knowledge of PyTorch and machine learning frameworks

Preferred Qualifications:
- Experience with Docker and Kubernetes
- Familiarity with Apache Spark
- Knowledge of distributed computing

Responsibilities:
- Build and deploy machine learning models
- Collaborate with cross-functional teams
- Mentor junior data scientists

Education:
Bachelor's degree in Computer Science, Mathematics or related field
"""


def main():
    print("=" * 70)
    print("TESTING JOB ORCHESTRATOR — Full Pipeline")
    print("=" * 70)

    orchestrator = JobOrchestrator()

    # ----------------------------------------------------------------
    # 1. Full pipeline test
    # ----------------------------------------------------------------
    print("\n1. Running full pipeline on sample JD...")
    profile = orchestrator.process(SAMPLE_JD)

    print(f"\n   Title:               {profile.title}")
    print(f"   Company:             {profile.company}")
    print(f"   Location:            {profile.location}")
    print(f"   Required skills:     {profile.required_hard_skills}")
    print(f"   Nice-to-have:        {profile.nice_to_have_skills}")
    print(f"   Experience:          {profile.required_experience_years} years")
    print(f"   Education:           {profile.required_education[:60]}...")
    print(f"   Job embedding shape: {profile.job_embedding.shape}")
    print(f"   Skills embedded:     {list(profile.skills_embeddings.keys())}")

    checks = [
        (profile.title == "Senior Data Scientist",       "Title extracted"),
        (len(profile.required_hard_skills) >= 4,         "Required skills found"),
        (len(profile.nice_to_have_skills)  >= 2,         "Nice-to-have skills found"),
        (profile.required_experience_years == 5.0,       "Experience years correct"),
        (len(profile.required_education)   >  0,         "Education extracted"),
        (profile.job_embedding.shape       == (384,),    "Job embedding shape correct"),
        (len(profile.skills_embeddings)    >  0,         "Skills embeddings created"),
    ]

    print("\n   Checks:")
    all_passed = True
    for passed, label in checks:
        status = "✓" if passed else "⚠"
        print(f"   {status} {label}")
        if not passed:
            all_passed = False

    # ----------------------------------------------------------------
    # 2. Empty input edge case
    # ----------------------------------------------------------------
    print("\n2. Testing empty input...")
    empty_profile = orchestrator.process("")

    print(f"   title:          '{empty_profile.title}'")
    print(f"   required_skills: {empty_profile.required_hard_skills}")

    if empty_profile.title == "" and empty_profile.required_hard_skills == []:
        print("   ✓ Empty input returns empty JobProfile")
    else:
        print("   ⚠ Empty input not handled correctly")

    # ----------------------------------------------------------------
    # 3. Shared embeddings_creator test
    # ----------------------------------------------------------------
    print("\n3. Testing shared EmbeddingsCreator instance...")

    from src.modules.embeddings_creator import EmbeddingsCreator
    shared_embedder = EmbeddingsCreator()

    orc1 = JobOrchestrator(embeddings_creator=shared_embedder)
    orc2 = JobOrchestrator(embeddings_creator=shared_embedder)

    same_model = orc1.embedder is orc2.embedder
    print(f"   Same embedder instance: {same_model}")
    print(f"   {'✓' if same_model else '⚠'} Shared EmbeddingsCreator works correctly")

    # ----------------------------------------------------------------
    # 4. Summary
    # ----------------------------------------------------------------
    print("\n" + "=" * 70)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED — review ⚠ items above")
    print("=" * 70)


if __name__ == "__main__":
    main()
