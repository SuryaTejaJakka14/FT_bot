# tests/test_pipeline_scraper.py
"""
Tests for PipelineJobScraper.
Uses mock JobPosting objects — no network calls.
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from datetime import datetime, timezone

from src.modules.pipeline_config  import PipelineConfig
from src.modules.pipeline_scraper import PipelineJobScraper, ScrapedJob
from src.modules.job_scraper      import JobPosting
from src.modules.job_parser       import JobProfile


def check(passed: bool, label: str):
    print(f"   {'✓' if passed else '⚠'} {label}")


# ------------------------------------------------------------------
# Mock JobPosting objects
# ------------------------------------------------------------------

def make_posting(
    title="Senior ML Engineer",
    company="OpenAI",
    location="Remote",
    description="",
    requirements=None,
    url="https://linkedin.com/jobs/123",
    source="linkedin",
):
    return JobPosting(
        title        = title,
        company      = company,
        location     = location,
        job_type     = "Full-time",
        description  = description,
        requirements = requirements or [],
        url          = url,
        source       = source,
        created_at   = datetime(2026, 2, 22, tzinfo=timezone.utc),
    )


MOCK_POSTINGS = [
    make_posting(
        title       = "Senior ML Engineer",
        company     = "OpenAI",
        description = (
            "We require strong experience with python, tensorflow, and "
            "machine learning. SQL proficiency is required. "
            "Knowledge of docker is preferred."
        ),
        url    = "https://linkedin.com/jobs/123",
        source = "linkedin",
    ),
    make_posting(
        title       = "Data Scientist",
        company     = "Google",
        description = (
            "Must have expertise in python and pandas. "
            "Experience with pytorch is a plus. SQL skills necessary."
        ),
        url    = "https://indeed.com/jobs/456",
        source = "indeed",
    ),
    make_posting(
        title        = "ML Research Engineer",
        company      = "DeepMind",
        description  = "",
        requirements = ["python", "pytorch", "deep learning"],
        url          = "https://linkedin.com/jobs/789",
        source       = "linkedin",
    ),
]


def main():
    print("=" * 70)
    print("TESTING PIPELINE JOB SCRAPER")
    print("=" * 70)

    config  = PipelineConfig(
        search_role = "ML Engineer",
        resume_path = "data/resumes/my_resume.pdf",
    )
    scraper = PipelineJobScraper(config, enable_selenium=False)

    # ----------------------------------------------------------------
    # 1. Core conversion
    # ----------------------------------------------------------------
    print("\n1. Testing scrape_from_postings()...")

    jobs = scraper.scrape_from_postings(MOCK_POSTINGS)

    print(f"\n   Converted {len(jobs)} jobs:\n")
    for j in jobs:
        print(f"   [{j.source}] {j.raw_title} @ {j.company}")
        print(f"     job_id   = {j.job_id}")
        print(f"     required = {j.profile.required_hard_skills}")
        print(f"     nice     = {j.profile.nice_to_have_skills}")

    print()
    checks = [
        (len(jobs) == 3,
            "Returns 3 jobs"),
        (all(isinstance(j, ScrapedJob) for j in jobs),
            "All are ScrapedJob"),
        (all(isinstance(j.profile, JobProfile) for j in jobs),
            "All have JobProfile"),
        (all(j.job_id != "" for j in jobs),
            "All have job_id"),
        (jobs[0].raw_title == "Senior ML Engineer",
            "First title correct"),
        (jobs[0].company   == "OpenAI",
            "First company correct"),
        (jobs[0].source    == "linkedin",
            "Source preserved"),
        (jobs[0].date_posted == "2026-02-22",
            "Date preserved"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 2. Skill extraction from description
    # ----------------------------------------------------------------
    print("\n2. Testing skill extraction from description...")

    j0 = jobs[0]   # "require python, tensorflow, ML; docker preferred"
    print(f"\n   {j0.raw_title} @ {j0.company}:")
    print(f"   required = {j0.profile.required_hard_skills}")
    print(f"   nice     = {j0.profile.nice_to_have_skills}")

    print()
    checks = [
        ("python"           in j0.profile.required_hard_skills,
            "python → required"),
        ("tensorflow"       in j0.profile.required_hard_skills,
            "tensorflow → required"),
        ("machine learning" in j0.profile.required_hard_skills,
            "machine learning → required"),
        ("docker"           in j0.profile.nice_to_have_skills,
            "docker → nice to have"),
        ("docker" not in j0.profile.required_hard_skills,
            "docker NOT in required"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 3. Requirements list takes priority over description extraction
    # ----------------------------------------------------------------
    print("\n3. Testing requirements list takes priority...")

    j2 = jobs[2]   # has requirements=["python", "pytorch", "deep learning"]
    print(f"\n   {j2.raw_title} @ {j2.company}:")
    print(f"   required = {j2.profile.required_hard_skills}")

    print()
    checks = [
        ("python"        in j2.profile.required_hard_skills,
            "python from requirements list"),
        ("pytorch"       in j2.profile.required_hard_skills,
            "pytorch from requirements list"),
        ("deep learning" in j2.profile.required_hard_skills,
            "deep learning from requirements list"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 4. JobProfile structure
    # ----------------------------------------------------------------
    print("\n4. Testing JobProfile structure...")

    p = jobs[0].profile
    print(f"\n   title    = {p.title}")
    print(f"   company  = {p.company}")
    print(f"   embedding shape = {p.job_embedding.shape}")

    print()
    checks = [
        (p.title    == "Senior ML Engineer",  "title set"),
        (p.company  == "OpenAI",              "company set"),
        (p.location == "Remote",              "location set"),
        (p.job_embedding.shape == (384,),     "embedding is 384-dim"),
        (len(p.skills_embeddings) > 0,        "skills_embeddings populated"),
        (p.version  == "1.0",                 "version set"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 5. Deduplication key
    # ----------------------------------------------------------------
    print("\n5. Testing deduplication key...")

    id_a = scraper._make_job_id(
        "Senior ML Engineer", "OpenAI", "https://linkedin.com/jobs/123"
    )
    id_b = scraper._make_job_id(
        "Senior ML Engineer", "OpenAI", "https://linkedin.com/jobs/123"
    )
    id_c = scraper._make_job_id(
        "Data Scientist", "Google", "https://indeed.com/jobs/456"
    )

    print(f"\n   id_a = {id_a}  (OpenAI)")
    print(f"   id_b = {id_b}  (OpenAI — same)")
    print(f"   id_c = {id_c}  (Google — different)")

    print()
    checks = [
        (id_a == id_b,    "Same job → same id"),
        (id_a != id_c,    "Different job → different id"),
        (len(id_a) == 16, "id is 16 chars"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 6. Edge cases
    # ----------------------------------------------------------------
    print("\n6. Testing edge cases...")

    # Empty description + no requirements → fallback to search_role
    empty = make_posting(
        description  = "",
        requirements = [],
        url          = "http://test.com/empty",
    )
    job_empty = scraper._convert(empty)
    print(f"\n   Empty description fallback: "
          f"{job_empty.profile.required_hard_skills}")
    check(
        len(job_empty.profile.required_hard_skills) > 0,
        "Empty description → fallback to search_role"
    )

    # Empty list
    empty_result = scraper.scrape_from_postings([])
    check(empty_result == [], "Empty postings → []")

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

