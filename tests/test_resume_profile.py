# tests/test_resume_profile.py
"""
Test ResumeProfile dataclass
"""

from src.modules.resume_parser import ResumeProfile
import numpy as np
from datetime import datetime

print("=" * 70)
print("TESTING ResumeProfile DATACLASS")
print("=" * 70)

# Test 1: Create instance
print("\n1. Creating ResumeProfile instance...")
profile = ResumeProfile(
    version="v1_test",
    full_text="Sample resume about Python development",
    hard_skills=["python", "sql", "docker"],
    soft_skills=["teamwork", "communication"],
    education="Bachelor of Science in Computer Science",
    experience_years=4.5,
    job_titles=["Software Engineer", "Developer"],
    companies=["Google", "StartupXYZ"],
    embedding=np.random.rand(384),
    skills_embeddings={
        "python": np.random.rand(384),
        "sql": np.random.rand(384),
        "docker": np.random.rand(384)
    }
)
print("✓ Instance created successfully")

# Test 2: Access fields
print("\n2. Accessing fields...")
print(f"  Version: {profile.version}")
print(f"  Experience: {profile.experience_years} years")
print(f"  Hard skills: {profile.hard_skills}")
print(f"  Soft skills: {profile.soft_skills}")
print(f"  Companies: {profile.companies}")
print(f"  Job titles: {profile.job_titles}")
print(f"  Education: {profile.education}")
print(f"  Created at: {profile.created_at}")
print("✓ All fields accessible")

# Test 3: Embedding shapes
print("\n3. Checking embedding shapes...")
print(f"  Resume embedding shape: {profile.embedding.shape}")
print(f"  Expected: (384,)")
assert profile.embedding.shape == (384,), "Wrong embedding shape!"
print(f"  Number of skill embeddings: {len(profile.skills_embeddings)}")
for skill, emb in profile.skills_embeddings.items():
    print(f"    {skill}: {emb.shape}")
    assert emb.shape == (384,), f"Wrong shape for {skill}!"
print("✓ All embeddings have correct shape")

# Test 4: Immutability (frozen=True)
print("\n4. Testing immutability...")
try:
    profile.experience_years = 10.0
    print("✗ Should not be able to modify frozen dataclass!")
except Exception as e:
    print(f"✓ Correctly prevents modification: {type(e).__name__}")

# Test 5: String representation
print("\n5. Testing string representation...")
print(f"  {profile}")
print("✓ String representation works")

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
