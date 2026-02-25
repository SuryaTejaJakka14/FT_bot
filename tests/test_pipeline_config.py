# tests/test_pipeline_config.py
"""
Tests for PipelineConfig.
Verifies field defaults, validation, and convenience methods.
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.modules.pipeline_config import PipelineConfig


def check(passed: bool, label: str):
    print(f"   {'✓' if passed else '⚠'} {label}")


def main():
    print("=" * 70)
    print("TESTING PIPELINE CONFIG")
    print("=" * 70)

    # ----------------------------------------------------------------
    # 1. Minimal construction (required fields only)
    # ----------------------------------------------------------------
    print("\n1. Testing minimal construction...")

    config = PipelineConfig(
        search_role = "ML Engineer",
        resume_path = "data/resumes/my_resume.pdf",
    )

    print(f"\n   {config.summary()}\n")

    checks = [
        (config.search_role     == "ML Engineer",          "search_role set correctly"),
        (config.location        == "Remote",               "location defaults to Remote"),
        (config.remote_only     == True,                   "remote_only defaults to True"),
        (config.results_wanted  == 20,                     "results_wanted defaults to 20"),
        (config.hours_old       == 72,                     "hours_old defaults to 72"),
        (config.min_score       == 0.50,                   "min_score defaults to 0.50"),
        (config.top_n           == 20,                     "top_n defaults to 20"),
        (config.db_path         == "data/jobs.db",         "db_path defaults correctly"),
        ("linkedin" in config.job_sites,                   "linkedin in default job_sites"),
        ("indeed"   in config.job_sites,                   "indeed in default job_sites"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 2. Custom construction
    # ----------------------------------------------------------------
    print("\n2. Testing custom construction...")

    custom = PipelineConfig(
        search_role    = "Data Scientist",
        location       = "New York, NY",
        remote_only    = False,
        results_wanted = 30,
        hours_old      = 48,
        job_sites      = ["linkedin"],
        min_score      = 0.65,
        top_n          = 10,
        resume_path    = "data/resumes/resume.pdf",
        db_path        = "data/custom.db",
    )

    checks = [
        (custom.search_role    == "Data Scientist",   "search_role set"),
        (custom.location       == "New York, NY",     "location set"),
        (custom.remote_only    == False,              "remote_only set"),
        (custom.results_wanted == 30,                 "results_wanted set"),
        (custom.hours_old      == 48,                 "hours_old set"),
        (custom.job_sites      == ["linkedin"],       "job_sites set"),
        (custom.min_score      == 0.65,               "min_score set"),
        (custom.top_n          == 10,                 "top_n set"),
        (custom.db_path        == "data/custom.db",   "db_path set"),
    ]
    print()
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 3. Weights validation
    # ----------------------------------------------------------------
    print("\n3. Testing weights validation...")

    default_weights = config.weights
    weight_sum      = sum(default_weights.values())
    print(f"\n   Default weights: {default_weights}")
    print(f"   Sum: {weight_sum:.4f}\n")

    checks = [
        (abs(weight_sum - 1.0) < 0.001,              "Default weights sum to 1.0"),
        ("semantic"   in default_weights,             "Has semantic key"),
        ("skills"     in default_weights,             "Has skills key"),
        ("experience" in default_weights,             "Has experience key"),
        ("education"  in default_weights,             "Has education key"),
    ]
    for passed, label in checks:
        check(passed, label)

    # ----------------------------------------------------------------
    # 4. Each config instance gets its own list (no shared state)
    # ----------------------------------------------------------------
    print("\n4. Testing independent defaults (no shared mutable state)...")

    config_a = PipelineConfig(search_role="Role A", resume_path="a.pdf")
    config_b = PipelineConfig(search_role="Role B", resume_path="b.pdf")
    config_a.job_sites.append("glassdoor")

    print(f"\n   config_a.job_sites: {config_a.job_sites}")
    print(f"   config_b.job_sites: {config_b.job_sites}\n")
    check(
        "glassdoor" not in config_b.job_sites,
        "Modifying config_a.job_sites does not affect config_b"
    )

    # ----------------------------------------------------------------
    # 5. Validation — catches bad values
    # ----------------------------------------------------------------
    print("\n5. Testing validation catches bad values...")

    bad_cases = [
        ({"search_role": "X", "resume_path": "r.pdf", "min_score": 1.5},
         "min_score > 1.0"),
        ({"search_role": "X", "resume_path": "r.pdf", "min_score": -0.1},
         "min_score < 0.0"),
        ({"search_role": "X", "resume_path": "r.pdf", "top_n": 0},
         "top_n = 0"),
        ({"search_role": "X", "resume_path": "r.pdf", "results_wanted": -1},
         "results_wanted negative"),
        ({"search_role": "X", "resume_path": "r.pdf", "job_sites": []},
         "empty job_sites"),
        ({"search_role": "X", "resume_path": "r.pdf",
          "job_sites": ["fakesite"]},
         "unsupported job site"),
        ({"search_role": "X", "resume_path": "r.pdf",
          "weights": {"semantic": 0.5, "skills": 0.5,
                      "experience": 0.5, "education": 0.5}},
         "weights do not sum to 1.0"),
    ]

    print()
    for kwargs, label in bad_cases:
        try:
            PipelineConfig(**kwargs)
            print(f"   ⚠ Should have raised ValueError: {label}")
        except ValueError:
            print(f"   ✓ Correctly rejects: {label}")

    # ----------------------------------------------------------------
    # 6. to_dict()
    # ----------------------------------------------------------------
    print("\n6. Testing to_dict()...")

    d = config.to_dict()
    print(f"\n   Keys: {list(d.keys())}\n")
    checks = [
        ("search_role"    in d, "search_role in dict"),
        ("min_score"      in d, "min_score in dict"),
        ("weights"        in d, "weights in dict"),
        ("db_path"        in d, "db_path in dict"),
        (d["search_role"] == "ML Engineer", "search_role value correct"),
    ]
    for passed, label in checks:
        check(passed, label)

    print("\n" + "=" * 70)
    print("ALL TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
