# tests/test_resume_orchestrator.py
"""
End-to-end test for ResumeOrchestrator.
"""

import sys
import os
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.resume_orchestrator import ResumeOrchestrator  # noqa: E402
from src.modules.profile_serializer import ProfileSerializer  # noqa: E402


def main():
    print("=" * 70)
    print("TESTING RESUME ORCHESTRATOR (END-TO-END)")
    print("=" * 70)

    orchestrator = ResumeOrchestrator()
    serializer = ProfileSerializer()

    # 1. Choose one of your parsed resume texts (or PDF)
    # Option A: Start from PDF
    pdf_path = "data/resumes/Surya_Teja_Jakka_Data.pdf"

    if not Path(pdf_path).exists():
        print(f"\n⚠ PDF not found at {pdf_path}")
        print("   Please adjust pdf_path in test_resume_orchestrator.py to a real file.")
        return

    print("\n1. Processing PDF resume...")
    profile = orchestrator.process_pdf(pdf_path, profile_version="v1_data_full")

    print("\n   Extracted profile:")
    print(f"   Version:          {profile.version}")
    print(f"   Hard skills:      {profile.hard_skills}")
    print(f"   Soft skills:      {profile.soft_skills}")
    print(f"   Education:        {profile.education}")
    print(f"   Embedding shape:  {profile.resume_embedding.shape}")
    print(f"   Skill emb count:  {len(profile.skills_embeddings)}")

    # 2. Save profile to JSON
    print("\n2. Saving profile JSON...")
    output_dir = Path("data/profiles")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "Surya_Teja_Jakka_Data_full.json")

    saved_path = orchestrator.save_profile_json(profile, output_path)
    print(f"   Saved to: {saved_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024:.1f} KB")

    # 3. Reload profile from JSON and confirm it looks OK
    print("\n3. Reloading profile from JSON...")
    loaded = serializer.load_profile(output_path)

    print(f"   Version:          {loaded.version}")
    print(f"   Hard skills:      {loaded.hard_skills}")
    print(f"   Soft skills:      {loaded.soft_skills}")
    print(f"   Education:        {loaded.education}")
    print(f"   Embedding shape:  {loaded.resume_embedding.shape}")
    print(f"   Skill emb count:  {len(loaded.skills_embeddings)}")

    print("\n   Basic checks:")
    print(f"   - Hard skills preserved: {loaded.hard_skills == profile.hard_skills}")
    print(f"   - Soft skills preserved: {loaded.soft_skills == profile.soft_skills}")
    print(f"   - Education preserved:   {loaded.education == profile.education}")
    print(f"   - Embedding shape OK:    {loaded.resume_embedding.shape == profile.resume_embedding.shape}")

    print("\n" + "=" * 70)
    print("END-TO-END ORCHESTRATION TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
