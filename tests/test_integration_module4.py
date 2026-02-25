# tests/test_integration_module4.py
"""
Module 4 Integration Test — uses the REAL MatchingEngine.

Builds proper ResumeProfile and JobProfile objects with random
unit-vector embeddings. Exact skill matching works correctly
regardless of embedding values.

Jobs and resumes are passed as wrapper dicts:
    {"job_id": "...",    "profile": JobProfile}
    {"resume_id": "...", "profile": ResumeProfile}

This lets the rankers extract the string id from the dict
while passing the typed profile object to the matcher.

Run with:
    python tests/test_integration_module4.py
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import numpy as np
from datetime import datetime

from src.modules.resume_parser   import ResumeProfile
from src.modules.job_parser      import JobProfile
from src.modules.matching_engine import MatchingEngine
from src.modules.ranking_engine  import RankingEngine


# ─────────────────────────────────────────────────────────────────────
# PROFILE BUILDERS
# ─────────────────────────────────────────────────────────────────────

EMBEDDING_DIM = 384   # standard sentence-transformer dimension


def _unit_vec(seed: int) -> np.ndarray:
    """Reproducible random unit vector."""
    rng = np.random.RandomState(seed)
    v   = rng.randn(EMBEDDING_DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


def make_resume(
    resume_id:   str,
    hard_skills: list,
    soft_skills: list  = None,
    education:   str   = "Bachelor's in Computer Science",
    years_exp:   float = 3.0,
    seed:        int   = 0,
) -> ResumeProfile:
    skills = hard_skills + (soft_skills or [])
    return ResumeProfile(
        version                = "1.0",
        hard_skills            = hard_skills,
        soft_skills            = soft_skills or [],
        education              = education,
        total_experience_years = years_exp,
        job_history            = [f"Engineer at Company ({int(years_exp)} yrs)"],
        resume_embedding       = _unit_vec(seed),
        skills_embeddings      = {s: _unit_vec(seed + i + 1)
                                  for i, s in enumerate(skills)},
        raw_text               = f"{resume_id} " + " ".join(skills),
        created_at             = datetime.now(),
    )


def make_job(
    job_id:          str,
    title:           str,
    required_skills: list,
    nice_to_have:    list  = None,
    years_req:       float = 2.0,
    seed:            int   = 200,
) -> JobProfile:
    all_skills = required_skills + (nice_to_have or [])
    return JobProfile(
        version                   = "1.0",
        title                     = title,
        company                   = "",
        location                  = "",
        required_hard_skills      = required_skills,
        nice_to_have_skills       = nice_to_have or [],
        required_experience_years = years_req,
        required_education        = "Bachelor's",
        job_embedding             = _unit_vec(seed),
        skills_embeddings         = {s: _unit_vec(seed + i + 1)
                                     for i, s in enumerate(all_skills)},
        created_at                = datetime.now(),
    )


# ─────────────────────────────────────────────────────────────────────
# TEST DATA
# profiles wrapped in dicts so rankers can extract ids
# ─────────────────────────────────────────────────────────────────────

# Fixed resume — passed directly to engine (candidate view)
ALICE_RESUME = make_resume(
    "alice",
    hard_skills=["python", "machine learning", "sql", "tensorflow", "pandas"],
    years_exp=4.0,
    seed=1,
)

# Jobs wrapped: {"job_id": ..., "profile": JobProfile}
JOBS = [
    {"job_id": "ml_engineer",
     "profile": make_job("ml_engineer", "ML Engineer",
                         ["python", "machine learning", "tensorflow"],
                         ["sql", "pytorch"], years_req=3.0, seed=201)},

    {"job_id": "data_analyst",
     "profile": make_job("data_analyst", "Data Analyst",
                         ["sql", "python", "pandas"],
                         ["tableau", "excel"], years_req=2.0, seed=202)},

    {"job_id": "backend_engineer",
     "profile": make_job("backend_engineer", "Backend Engineer",
                         ["python", "docker", "postgres"],
                         ["kubernetes", "redis"], years_req=2.0, seed=203)},

    {"job_id": "devops_engineer",
     "profile": make_job("devops_engineer", "DevOps Engineer",
                         ["docker", "kubernetes", "terraform"],
                         ["python", "ansible"], years_req=3.0, seed=204)},
]

# Fixed job — passed directly to engine (recruiter view)
ML_JOB = make_job(
    "ml_engineer", "ML Engineer",
    ["python", "machine learning", "tensorflow"],
    ["sql", "pytorch"], years_req=3.0, seed=201,
)

# Candidates wrapped: {"resume_id": ..., "profile": ResumeProfile}
CANDIDATES = [
    {"resume_id": "alice",
     "profile": make_resume("alice",
                            ["python", "machine learning", "tensorflow",
                             "sql", "pandas"],
                            years_exp=4.0, seed=1)},

    {"resume_id": "bob",
     "profile": make_resume("bob",
                            ["python", "machine learning", "sql"],
                            years_exp=3.0, seed=2)},

    {"resume_id": "charlie",
     "profile": make_resume("charlie",
                            ["python", "docker", "postgres", "redis"],
                            years_exp=3.0, seed=3)},

    {"resume_id": "diana",
     "profile": make_resume("diana",
                            ["sql", "tableau", "excel", "pandas"],
                            years_exp=2.0, seed=4)},

    {"resume_id": "eve",
     "profile": make_resume("eve",
                            ["java", "spring", "hibernate"],
                            years_exp=2.0, seed=5)},
]


# ─────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────

def section(title):
    print(f"\n{title}")

def check(passed: bool, label: str):
    print(f"   {'✓' if passed else '⚠'} {label}")


# ─────────────────────────────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("MODULE 4 INTEGRATION TEST — Real MatchingEngine + RankingEngine")
    print("=" * 70)

    matcher = MatchingEngine()
    engine  = RankingEngine(matcher)

    # ──────────────────────────────────────────────────────────────────
    # 1. Smoke test: real MatchingEngine produces a valid MatchResult
    # ──────────────────────────────────────────────────────────────────
    section("1. Smoke test — real MatchingEngine works...")

    match = matcher.match(ALICE_RESUME, ML_JOB)

    print(f"\n   alice vs ml_engineer:")
    print(f"   overall_score  = {match.overall_score:.4f}")
    print(f"   matched_skills = {match.matched_skills}")
    print(f"   missing_skills = {match.missing_skills}")

    print()
    for passed, label in [
        (0.0 <= match.overall_score <= 1.0, "overall_score in [0, 1]"),
        (hasattr(match, "matched_skills"),   "has matched_skills"),
        (hasattr(match, "missing_skills"),   "has missing_skills"),
        (hasattr(match, "overall_score"),    "has overall_score"),
    ]:
        check(passed, label)

    # ──────────────────────────────────────────────────────────────────
    # 2. rank_jobs_for_resume() — candidate view
    # ──────────────────────────────────────────────────────────────────
    section("2. rank_jobs_for_resume() — candidate view...")

    job_results = engine.rank_jobs_for_resume(ALICE_RESUME, JOBS)

    print(f"\n   Alice's best-fit jobs (ranked):")
    print(f"   {'Rank':<5} {'Job':<22} {'Score':>7}  {'Normalized':>11}  "
          f"{'Percentile':>11}  {'Label'}")
    print(f"   {'-'*5} {'-'*22} {'-'*7}  {'-'*11}  {'-'*11}  {'-'*12}")
    for r in job_results:
        print(f"   #{r.rank:<4} {r.job_id:<22} {r.overall_score:>7.4f}  "
              f"{r.normalized_score:>11.4f}  {r.percentile:>11.4f}  {r.rank_label}")

    ml_r     = next(r for r in job_results if r.job_id == "ml_engineer")
    devops_r = next(r for r in job_results if r.job_id == "devops_engineer")

    print()
    for passed, label in [
        (len(job_results) == 4,
            "Returns all 4 jobs"),
        (job_results[0].rank == 1,
            "First result has rank 1"),
        (job_results[-1].rank == 4,
            "Last result has rank 4"),
        (job_results[0].normalized_score == 1.0,
            "Best → normalized 1.0"),
        (job_results[-1].normalized_score == 0.0,
            "Worst → normalized 0.0"),
        (all(0.0 <= r.overall_score <= 1.0 for r in job_results),
            "All scores in [0, 1]"),
        (all(r.match_result is not None for r in job_results),
            "All have match_result"),
        (all(r.rank_label != "" for r in job_results),
            "All have rank labels"),
        (ml_r.overall_score > devops_r.overall_score,
            "ml_engineer scores higher than devops_engineer for Alice"),
    ]:
        check(passed, label)

    # ──────────────────────────────────────────────────────────────────
    # 3. rank_resumes_for_job() — recruiter view
    # ──────────────────────────────────────────────────────────────────
    section("3. rank_resumes_for_job() — recruiter view...")

    resume_results = engine.rank_resumes_for_job(ML_JOB, CANDIDATES)

    print(f"\n   Best candidates for ml_engineer (ranked):")
    print(f"   {'Rank':<5} {'Candidate':<10} {'Score':>7}  {'Normalized':>11}  "
          f"{'Percentile':>11}  {'Label'}")
    print(f"   {'-'*5} {'-'*10} {'-'*7}  {'-'*11}  {'-'*11}  {'-'*12}")
    for r in resume_results:
        print(f"   #{r.rank:<4} {r.resume_id:<10} {r.overall_score:>7.4f}  "
              f"{r.normalized_score:>11.4f}  {r.percentile:>11.4f}  {r.rank_label}")

    alice_r = next(r for r in resume_results if r.resume_id == "alice")
    eve_r   = next(r for r in resume_results if r.resume_id == "eve")

    print()
    for passed, label in [
        (len(resume_results) == 5,
            "Returns all 5 candidates"),
        (resume_results[0].rank == 1,
            "First result has rank 1"),
        (resume_results[0].normalized_score == 1.0,
            "Best → normalized 1.0"),
        (resume_results[-1].normalized_score == 0.0,
            "Worst → normalized 0.0"),
        (all(r.match_result is not None for r in resume_results),
            "All have match_result"),
        (alice_r.overall_score > eve_r.overall_score,
            "Alice scores higher than Eve for ML job"),
    ]:
        check(passed, label)

    # ──────────────────────────────────────────────────────────────────
    # 4. shortlist_resumes()
    # ──────────────────────────────────────────────────────────────────
    section("4. shortlist_resumes()...")

    threshold = 0.30
    shortlist  = engine.shortlist_resumes(ML_JOB, CANDIDATES,
                                          top_n=3, min_score=threshold)

    print(f"\n   top_n=3, min_score={threshold}:")
    for r in shortlist:
        print(f"   #{r.rank} {r.resume_id:<10} score={r.overall_score:.4f}")

    print()
    for passed, label in [
        (len(shortlist) <= 3,
            "Returns at most 3"),
        (all(r.overall_score >= threshold for r in shortlist),
            f"All scores >= {threshold}"),
        (shortlist[0].resume_id == resume_results[0].resume_id if shortlist else True,
            "Shortlist leader matches ranked leader"),
    ]:
        check(passed, label)

    # ──────────────────────────────────────────────────────────────────
    # 5. match_result passthrough — Module 3 data preserved
    # ──────────────────────────────────────────────────────────────────
    section("5. match_result passthrough — Module 3 data intact...")

    top = resume_results[0]

    print(f"\n   Top result ({top.resume_id}) match_result fields:")
    print(f"   overall_score  = {top.match_result.overall_score:.4f}")
    print(f"   matched_skills = {top.match_result.matched_skills}")
    print(f"   missing_skills = {top.match_result.missing_skills}")

    print()
    for passed, label in [
        (top.match_result is not None,
            "match_result exists"),
        (top.overall_score == top.match_result.overall_score,
            "RankingResult.overall_score == match_result.overall_score"),
        (isinstance(top.match_result.matched_skills, list),
            "matched_skills is a list"),
        (isinstance(top.match_result.missing_skills, list),
            "missing_skills is a list"),
    ]:
        check(passed, label)

    # ──────────────────────────────────────────────────────────────────
    # 6. get_stats()
    # ──────────────────────────────────────────────────────────────────
    section("6. get_stats() diagnostics...")

    stats = engine.get_stats("resumes", ML_JOB, CANDIDATES)
    print(f"\n   Recruiter pool stats (5 candidates for ml_engineer):")
    for key in ["n", "best_candidate", "best_score", "worst_score",
                "score_range", "mean_score"]:
        print(f"   {key:<16}: {stats[key]}")

    print()
    for passed, label in [
        (stats["n"]           == 5,                    "n = 5"),
        (stats["best_score"]  >= stats["worst_score"], "best >= worst"),
        (stats["score_range"] >= 0.0,                  "score_range >= 0"),
        (0.0 <= stats["mean_score"] <= 1.0,            "mean_score in [0, 1]"),
    ]:
        check(passed, label)

    print("\n" + "=" * 70)
    print("ALL INTEGRATION TESTS COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
