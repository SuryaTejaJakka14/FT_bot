# tests/test_profile_serializer.py
"""
Test ProfileSerializer (aligned with ResumeProfile fields).
"""

import sys
import os
from datetime import datetime
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.modules.profile_serializer import ProfileSerializer  # noqa: E402
from src.modules.resume_parser import ResumeProfile  # noqa: E402


def main():
    print("=" * 70)
    print("TESTING PROFILE SERIALIZER")
    print("=" * 70)

    serializer = ProfileSerializer()

    # 1. Create a sample ResumeProfile
    print("\n1. Creating sample ResumeProfile...")

    sample_embedding = np.random.randn(384).astype(np.float32)
    sample_skills_embs = {
        "python": np.random.randn(384).astype(np.float32),
        "tensorflow": np.random.randn(384).astype(np.float32),
        "aws": np.random.randn(384).astype(np.float32),
    }

    profile = ResumeProfile(
        version="v1_data_scientist",
        hard_skills=["python", "tensorflow", "aws", "docker", "sql"],
        soft_skills=["leadership", "communication", "teamwork"],
        education="Master in Computer Science | Stanford University",
        resume_embedding=sample_embedding,
        skills_embeddings=sample_skills_embs,
        created_at=datetime.now(),
    )

    print(f"   Version: {profile.version}")
    print(f"   Hard skills: {profile.hard_skills}")
    print(f"   Soft skills: {profile.soft_skills}")
    print(f"   Education: {profile.education}")
    print(f"   Embedding shape: {profile.resume_embedding.shape}")
    print(f"   Skills embeddings: {len(profile.skills_embeddings)}")
    print("   ✓ Profile created")

    # 2. Save profile to JSON
    print("\n2. Saving profile to JSON...")

    output_dir = Path("data/profiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "test_profile.json")

    saved_path = serializer.save_profile(profile, output_path)
    print(f"   Saved to: {saved_path}")

    file_size = Path(output_path).stat().st_size / 1024
    print(f"   File size: {file_size:.1f} KB")
    print("   ✓ Profile saved")

    # 3. Load profile from JSON
    print("\n3. Loading profile from JSON...")

    loaded = serializer.load_profile(output_path)
    print(f"   Version: {loaded.version}")
    print(f"   Hard skills: {loaded.hard_skills}")
    print(f"   Soft skills: {loaded.soft_skills}")
    print(f"   Education: {loaded.education}")
    print(f"   Embedding shape: {loaded.resume_embedding.shape}")
    print(f"   Skills embeddings: {len(loaded.skills_embeddings)}")
    print("   ✓ Profile loaded")

    # 4. Verify data integrity
    print("\n4. Verifying data integrity...")

    checks = {
        "version": profile.version == loaded.version,
        "hard_skills": profile.hard_skills == loaded.hard_skills,
        "soft_skills": profile.soft_skills == loaded.soft_skills,
        "education": profile.education == loaded.education,
        "embedding_shape": profile.resume_embedding.shape == loaded.resume_embedding.shape,
        "embedding_values": np.allclose(profile.resume_embedding, loaded.resume_embedding),
        "skills_emb_keys": set(profile.skills_embeddings.keys()) == set(loaded.skills_embeddings.keys()),
    }

    for skill in profile.skills_embeddings:
        key = f"skill_emb_{skill}"
        checks[key] = np.allclose(
            profile.skills_embeddings[skill],
            loaded.skills_embeddings[skill],
        )

    all_passed = True
    for name, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"   {status} {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n   ✅ ALL DATA INTEGRITY CHECKS PASSED!")
    else:
        print("\n   ⚠ Some checks failed — see above")

    # 5. Test profile_exists
    print("\n5. Testing profile_exists...")
    exists = serializer.profile_exists(output_path)
    print(f"   '{output_path}' exists: {exists}")
    if exists:
        print("   ✓ Correctly detected existing file")

    # 6. Inspect JSON structure
    print("\n6. Inspecting saved JSON structure...")

    with open(output_path, "r", encoding="utf-8") as f:
        raw_json = f.read()

    print("   First 400 characters of JSON:")
    print("   " + "-" * 60)
    preview = raw_json[:400].replace("\n", "\n   ")
    print(f"   {preview}")
    print("   " + "-" * 60)

    # 7. Cleanup
    print("\n7. Cleanup...")
    Path(output_path).unlink()
    print(f"   ✓ Deleted test file: {output_path}")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
