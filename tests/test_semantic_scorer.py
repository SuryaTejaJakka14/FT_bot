# tests/test_semantic_scorer.py
"""
Tests for SemanticScorer.
Verifies cosine similarity computation and score validation.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.modules.semantic_scorer import SemanticScorer
from src.modules.embeddings_creator import EmbeddingsCreator


def main():
    print("=" * 70)
    print("TESTING SEMANTIC SCORER")
    print("=" * 70)

    scorer  = SemanticScorer()
    creator = EmbeddingsCreator()

    # ----------------------------------------------------------------
    # 1. Basic cosine similarity with known vectors
    # ----------------------------------------------------------------
    print("\n1. Testing cosine similarity with known vectors...")

    # Identical vectors → score = 1.0
    vec = np.array([1.0, 0.0, 0.0])
    score = scorer.score(vec, vec)
    print(f"   Identical vectors:     {score:.4f}  (expected ~1.0)")
    print(f"   {'✓' if abs(score - 1.0) < 1e-6 else '⚠'} Identical vectors = 1.0")

    # Opposite vectors → score = -1.0 (clamped to 0.0)
    vec_a = np.array([1.0,  0.0, 0.0])
    vec_b = np.array([-1.0, 0.0, 0.0])
    score = scorer.score(vec_a, vec_b)
    print(f"   Opposite vectors:      {score:.4f}  (expected 0.0 after clamp)")
    print(f"   {'✓' if score == 0.0 else '⚠'} Opposite vectors clamped to 0.0")

    # Perpendicular vectors → score = 0.0
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 1.0, 0.0])
    score = scorer.score(vec_a, vec_b)
    print(f"   Perpendicular vectors: {score:.4f}  (expected 0.0)")
    print(f"   {'✓' if abs(score) < 1e-6 else '⚠'} Perpendicular vectors = 0.0")

    # Partially similar vectors
    vec_a = np.array([1.0, 1.0, 0.0])
    vec_b = np.array([1.0, 0.0, 0.0])
    score = scorer.score(vec_a, vec_b)
    expected = 1.0 / np.sqrt(2)  # ~0.707
    print(f"   Partially similar:     {score:.4f}  (expected ~{expected:.4f})")
    print(f"   {'✓' if abs(score - expected) < 1e-4 else '⚠'} Partial similarity correct")

    # ----------------------------------------------------------------
    # 2. score_with_details()
    # ----------------------------------------------------------------
    print("\n2. Testing score_with_details()...")

    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([1.0, 0.0, 0.0])
    details = scorer.score_with_details(vec_a, vec_b)

    print(f"   score:       {details['score']:.4f}")
    print(f"   raw_score:   {details['raw_score']:.4f}")
    print(f"   resume_norm: {details['resume_norm']:.4f}")
    print(f"   job_norm:    {details['job_norm']:.4f}")
    print(f"   dot_product: {details['dot_product']:.4f}")

    checks = [
        (abs(details["score"] - 1.0) < 1e-6,       "score = 1.0"),
        (abs(details["resume_norm"] - 1.0) < 1e-6,  "resume_norm = 1.0"),
        (abs(details["job_norm"] - 1.0) < 1e-6,     "job_norm = 1.0"),
        (abs(details["dot_product"] - 1.0) < 1e-6,  "dot_product = 1.0"),
    ]
    for passed, label in checks:
        print(f"   {'✓' if passed else '⚠'} {label}")

    # ----------------------------------------------------------------
    # 3. Real embedding similarity ranking
    # ----------------------------------------------------------------
    print("\n3. Testing semantic ranking with real embeddings...")

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
        "ML Engineer    (strong)": creator.create_text_embedding(
            creator.build_job_summary_text(
                title="Machine Learning Engineer",
                required_hard_skills=["python", "tensorflow", "pytorch", "aws"],
                nice_to_have_skills=["kubernetes"],
                required_experience_years=4.0,
                required_education="Master in Computer Science",
            )
        ),
        "Data Scientist (good)  ": creator.create_text_embedding(
            creator.build_job_summary_text(
                title="Senior Data Scientist",
                required_hard_skills=["python", "sql", "tensorflow"],
                nice_to_have_skills=["docker"],
                required_experience_years=5.0,
                required_education="Master in Computer Science",
            )
        ),
        "Frontend Dev   (weak)  ": creator.create_text_embedding(
            creator.build_job_summary_text(
                title="Frontend Developer",
                required_hard_skills=["javascript", "react", "css"],
                nice_to_have_skills=["typescript"],
                required_experience_years=3.0,
                required_education="Bachelor in Computer Science",
            )
        ),
        "Accountant     (none)  ": creator.create_text_embedding(
            creator.build_job_summary_text(
                title="Senior Accountant",
                required_hard_skills=["excel", "quickbooks", "gaap"],
                nice_to_have_skills=[],
                required_experience_years=3.0,
                required_education="Bachelor in Accounting",
            )
        ),
    }

    print(f"\n   Resume: Senior Data Scientist | Python, TensorFlow, AWS\n")
    print(f"   {'Job':<30s} Score  Bar")
    print(f"   {'-'*30} -----  ---")

    scored = []
    for job_title, job_emb in jobs.items():
        s = scorer.score(resume_emb, job_emb)
        scored.append((job_title, s))

    for title, s in sorted(scored, key=lambda x: x[1], reverse=True):
        bar = "█" * int(s * 30)
        print(f"   {title:<30s} {s:.3f}  {bar}")

    # Validate ranking
    sorted_titles = [t for t, _ in sorted(scored, key=lambda x: x[1], reverse=True)]
    tech_ranks  = [sorted_titles.index(t) for t in sorted_titles if "none" not in t.lower()]
    acc_rank    = sorted_titles.index("Accountant     (none)  ")
    fe_rank     = sorted_titles.index("Frontend Dev   (weak)  ")

    print(f"\n   {'✓' if acc_rank == 3 else '⚠'} Accountant ranked last")
    print(f"   {'✓' if fe_rank > 1 else '⚠'} Frontend Developer ranked below tech roles")

    # ----------------------------------------------------------------
    # 4. Edge cases
    # ----------------------------------------------------------------
    print("\n4. Testing edge cases...")

    # Zero vector (e.g. empty JobProfile embedding)
    zero_vec  = np.zeros(384)
    normal_vec = np.random.rand(384)
    score = scorer.score(zero_vec, normal_vec)
    print(f"   Zero vector score: {score:.4f}")
    print(f"   {'✓' if score == 0.0 else '⚠'} Zero vector → 0.0")

    # Shape mismatch
    try:
        scorer.score(np.zeros(384), np.zeros(128))
        print("   ⚠ Should have raised ValueError for shape mismatch")
    except ValueError as e:
        print(f"   ✓ Shape mismatch raises ValueError: '{e}'")

    # None input
    try:
        scorer.score(None, np.zeros(384))
        print("   ⚠ Should have raised ValueError for None input")
    except ValueError as e:
        print(f"   ✓ None input raises ValueError: '{e}'")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
