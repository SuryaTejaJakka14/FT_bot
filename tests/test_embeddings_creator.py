# tests/test_embeddings_creator.py
"""
Test EmbeddingsCreator
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.embeddings_creator import EmbeddingsCreator  # noqa: E402


def main():
    print("=" * 70)
    print("TESTING EMBEDDINGS CREATOR")
    print("=" * 70)

    creator = EmbeddingsCreator()

    # ------------------------------------------------------------------
    # 1. Model loading and info
    # ------------------------------------------------------------------
    print("\n1. Loading model and getting info...")
    info = creator.get_embedding_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    print("   ✓ Model loaded successfully")

    # ------------------------------------------------------------------
    # 2. Single text embedding
    # ------------------------------------------------------------------
    print("\n2. Testing single text embedding...")
    text = "Senior Python developer with 5 years of experience in machine learning"
    emb = creator.create_text_embedding(text)
    print(f"   Input: '{text[:50]}...'")
    print(f"   Shape: {emb.shape}")
    print(f"   Dtype: {emb.dtype}")
    print(f"   First 5 values: {emb[:5]}")

    if emb.shape == (384,):
        print("   ✓ Correct shape (384,)")
    else:
        print(f"   ⚠ Unexpected shape: {emb.shape}")

    # ------------------------------------------------------------------
    # 3. Empty text handling
    # ------------------------------------------------------------------
    print("\n3. Testing empty text handling...")
    empty_emb = creator.create_text_embedding("")
    print(f"   Empty text shape: {empty_emb.shape}")
    print(f"   Is zero vector: {(empty_emb == 0).all()}")

    if (empty_emb == 0).all():
        print("   ✓ Empty text returns zero vector")
    else:
        print("   ⚠ Empty text should return zero vector")

    # ------------------------------------------------------------------
    # 4. Batch embeddings
    # ------------------------------------------------------------------
    print("\n4. Testing batch embeddings...")
    texts = [
        "Python developer",
        "Java engineer",
        "Data scientist with TensorFlow",
        "Marketing manager",
    ]
    batch_embs = creator.create_batch_embeddings(texts)
    print(f"   Input: {len(texts)} texts")
    print(f"   Output shape: {batch_embs.shape}")

    if batch_embs.shape == (4, 384):
        print("   ✓ Correct batch shape (4, 384)")
    else:
        print(f"   ⚠ Unexpected shape: {batch_embs.shape}")

    # ------------------------------------------------------------------
    # 5. Skills embeddings
    # ------------------------------------------------------------------
    print("\n5. Testing skills embeddings...")
    skills = ["python", "tensorflow", "aws", "docker", "machine learning"]
    skills_embs = creator.create_skills_embeddings(skills)
    print(f"   Skills: {skills}")
    print(f"   Embeddings created: {len(skills_embs)}")

    for skill, emb in skills_embs.items():
        print(f"     {skill:20s} → shape {emb.shape}")

    if len(skills_embs) == len(skills):
        print("   ✓ All skills embedded")
    else:
        print("   ⚠ Missing skill embeddings")

    # ------------------------------------------------------------------
    # 6. Similarity: related vs unrelated texts
    # ------------------------------------------------------------------
    print("\n6. Testing similarity scores...")

    pairs = [
        ("Python developer", "Python programmer",
         "Very similar (same meaning)"),
        ("Python developer", "Java developer",
         "Somewhat similar (both developers)"),
        ("Python developer", "Machine learning engineer",
         "Moderately similar (tech roles)"),
        ("Python developer", "Professional chef",
         "Very different (unrelated fields)"),
    ]

    for text_a, text_b, description in pairs:
        emb_a = creator.create_text_embedding(text_a)
        emb_b = creator.create_text_embedding(text_b)
        sim = creator.compute_similarity(emb_a, emb_b)
        print(f"\n   '{text_a}' vs '{text_b}'")
        print(f"   {description}")
        print(f"   Similarity: {sim:.4f}")

    # ------------------------------------------------------------------
    # 7. Batch similarity (one resume vs multiple jobs)
    # ------------------------------------------------------------------
    print("\n7. Testing batch similarity (resume vs jobs)...")

    resume_text = "Senior Data Scientist with Python, TensorFlow, and AWS"
    resume_emb = creator.create_text_embedding(resume_text)

    jobs = [
        "Data Scientist with Python and machine learning experience",
        "Frontend Developer with React and CSS skills",
        "ML Engineer with TensorFlow and cloud deployment",
        "Accountant with Excel and financial modeling experience",
    ]

    job_embs = creator.create_batch_embeddings(jobs)
    scores = creator.compute_batch_similarity(resume_emb, job_embs)

    print(f"\n   Resume: '{resume_text}'")
    print(f"   {'Job Description':<60s} Score")
    print(f"   {'-'*60} -----")

    for job, score in sorted(zip(jobs, scores), key=lambda x: x[1], reverse=True):
        bar = "█" * int(score * 30)
        print(f"   {job:<60s} {score:.3f} {bar}")

    # ------------------------------------------------------------------
    # 8. Resume summary embedding
    # ------------------------------------------------------------------
    print("\n8. Testing resume summary builder...")

    summary = creator.build_resume_summary_text(
        hard_skills=["python", "tensorflow", "aws", "docker"],
        soft_skills=["leadership", "communication"],
        education="Master in Computer Science | Stanford University",
        experience_years=5.0,
        job_history=["Senior Data Scientist at Google (2021-Present)"],
    )
    print(f"   Summary: '{summary[:80]}...'")

    summary_emb = creator.create_text_embedding(summary)
    print(f"   Summary embedding shape: {summary_emb.shape}")
    print("   ✓ Resume summary embedded successfully")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
