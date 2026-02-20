# tests/test_job_profile.py
"""
Sanity test for JobProfile dataclass.
"""

import sys
import os
from datetime import datetime

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from src.modules.job_parser import JobProfile  # noqa: E402


def main():
    print("=" * 70)
    print("TESTING JOBPROFILE DATACLASS")
    print("=" * 70)

    emb = np.random.randn(384).astype(np.float32)
    skills_embs = {
        "python": np.random.randn(384).astype(np.float32),
        "tensorflow": np.random.randn(384).astype(np.float32),
    }

    profile = JobProfile(
        version="v1_data_scientist_job",
        title="Senior Data Scientist",
        company="Example Corp",
        location="Remote",
        required_hard_skills=["python", "tensorflow", "sql"],
        nice_to_have_skills=["docker", "kubernetes"],
        required_experience_years=5.0,
        required_education="Bachelor or Master in Computer Science or related field",
        job_embedding=emb,
        skills_embeddings=skills_embs,
        created_at=datetime.now(),
    )

    print(f"   Version:      {profile.version}")
    print(f"   Title:        {profile.title}")
    print(f"   Company:      {profile.company}")
    print(f"   Location:     {profile.location}")
    print(f"   Req skills:   {profile.required_hard_skills}")
    print(f"   Nice skills:  {profile.nice_to_have_skills}")
    print(f"   Req exp yrs:  {profile.required_experience_years}")
    print(f"   Education:    {profile.required_education}")
    print(f"   Emb shape:    {profile.job_embedding.shape}")
    print(f"   Skill emb n:  {len(profile.skills_embeddings)}")

    print("\n   ✓ JobProfile instantiated successfully")

    print("\n" + "=" * 70)
    print("JOBPROFILE TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
