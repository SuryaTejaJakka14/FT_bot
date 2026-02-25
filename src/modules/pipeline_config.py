# src/modules/pipeline_config.py
"""
PipelineConfig: Central configuration dataclass for Module 5.

Holds every tunable setting for the full job search pipeline:
  - What to search for   (role, location, sites)
  - How many results     (results_wanted, hours_old)
  - Quality thresholds   (min_score, top_n)
  - Score weights        (semantic, skills, experience, education)
  - File paths           (resume_path, db_path)

Usage:
    config = PipelineConfig(
        search_role  = "ML Engineer",
        location     = "Remote",
        resume_path  = "data/resumes/my_resume.pdf",
    )
"""

import logging
from dataclasses  import dataclass, field
from pathlib      import Path
from typing       import List, Optional

logger = logging.getLogger(__name__)


# Valid job sites supported by python-jobspy
SUPPORTED_SITES = {"linkedin", "indeed", "glassdoor", "zip_recruiter"}

# Valid application status values
APPLICATION_STATUSES = {
    "NOT_APPLIED",
    "APPLIED",
    "INTERVIEWING",
    "OFFER_RECEIVED",
    "REJECTED",
    "DECLINED",
}


@dataclass
class PipelineConfig:
    """
    Central configuration for the Module 5 pipeline.

    All fields have sensible defaults so you only need to set
    what differs from the defaults for your use case.

    Validation runs automatically via __post_init__.

    Required fields (no defaults):
        search_role:  str   — job title to search for
        resume_path:  str   — path to your resume file

    Example — minimal:
        config = PipelineConfig(
            search_role = "ML Engineer",
            resume_path = "data/resumes/my_resume.pdf",
        )

    Example — custom:
        config = PipelineConfig(
            search_role     = "Data Scientist",
            location        = "New York, NY",
            remote_only     = False,
            results_wanted  = 30,
            min_score       = 0.65,
            top_n           = 10,
            resume_path     = "data/resumes/my_resume.pdf",
            db_path         = "data/jobs.db",
        )
    """

    # ------------------------------------------------------------------
    # REQUIRED — no defaults
    # ------------------------------------------------------------------
    search_role: str                      # e.g. "ML Engineer"
    resume_path: str                      # e.g. "data/resumes/my_resume.pdf"

    # ------------------------------------------------------------------
    # SEARCH SETTINGS
    # ------------------------------------------------------------------
    location:        str       = "Remote"
    remote_only:     bool      = True
    results_wanted:  int       = 20       # jobs to fetch per source
    hours_old:       int       = 72       # only jobs posted within N hours
    job_sites: List[str]       = field(
        default_factory=lambda: ["linkedin", "indeed"]
    )

    # ------------------------------------------------------------------
    # MATCHING & RANKING THRESHOLDS
    # ------------------------------------------------------------------
    min_score:  float = 0.50      # minimum match score to save a job
    top_n:      int   = 20        # max jobs to display in tracker

    # ------------------------------------------------------------------
    # SCORE WEIGHTS  (must sum to 1.0)
    # ------------------------------------------------------------------
    weights: Optional[dict] = field(
        default_factory=lambda: {
            "semantic":   0.30,
            "skills":     0.50,
            "experience": 0.10,
            "education":  0.10,
        }
    )

    # ------------------------------------------------------------------
    # FILE PATHS
    # ------------------------------------------------------------------
    db_path: str = "data/jobs.db"

    # ------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------

    def __post_init__(self):
        """Validate all fields immediately after construction."""
        self._validate_scores()
        self._validate_counts()
        self._validate_sites()
        self._validate_weights()
        self._validate_paths()
        logger.info(
            f"PipelineConfig created: role='{self.search_role}', "
            f"location='{self.location}', min_score={self.min_score}, "
            f"sites={self.job_sites}"
        )

    def _validate_scores(self):
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(
                f"min_score must be in [0.0, 1.0], got {self.min_score}"
            )

    def _validate_counts(self):
        if self.top_n < 1:
            raise ValueError(f"top_n must be >= 1, got {self.top_n}")
        if self.results_wanted < 1:
            raise ValueError(
                f"results_wanted must be >= 1, got {self.results_wanted}"
            )
        if self.hours_old < 1:
            raise ValueError(f"hours_old must be >= 1, got {self.hours_old}")

    def _validate_sites(self):
        if not self.job_sites:
            raise ValueError("job_sites must not be empty")
        invalid = set(self.job_sites) - SUPPORTED_SITES
        if invalid:
            raise ValueError(
                f"Unsupported job sites: {invalid}. "
                f"Supported: {SUPPORTED_SITES}"
            )

    def _validate_weights(self):
        if self.weights is None:
            return
        required_keys = {"semantic", "skills", "experience", "education"}
        missing = required_keys - set(self.weights.keys())
        if missing:
            raise ValueError(f"weights missing keys: {missing}")
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            raise ValueError(
                f"weights must sum to 1.0, got {total:.4f}"
            )

    def _validate_paths(self):
        """Warn (don't crash) if resume file not found yet."""
        resume = Path(self.resume_path)
        if not resume.exists():
            logger.warning(
                f"Resume file not found: '{self.resume_path}'. "
                f"Ensure it exists before running the pipeline."
            )
        db_dir = Path(self.db_path).parent
        if not db_dir.exists():
            logger.warning(
                f"Database directory '{db_dir}' does not exist. "
                f"It will be created when JobStore initializes."
            )

    # ------------------------------------------------------------------
    # CONVENIENCE METHODS
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a one-line human-readable config summary."""
        remote = "Remote only" if self.remote_only else "Any location"
        sites  = ", ".join(self.job_sites)
        return (
            f"Role: '{self.search_role}' | "
            f"Location: {self.location} ({remote}) | "
            f"Sites: {sites} | "
            f"Results: {self.results_wanted} per site | "
            f"Min score: {self.min_score} | "
            f"Top N: {self.top_n}"
        )

    def to_dict(self) -> dict:
        """Serialise config to a plain dict (for logging or saving)."""
        return {
            "search_role":     self.search_role,
            "location":        self.location,
            "remote_only":     self.remote_only,
            "results_wanted":  self.results_wanted,
            "hours_old":       self.hours_old,
            "job_sites":       self.job_sites,
            "min_score":       self.min_score,
            "top_n":           self.top_n,
            "weights":         self.weights,
            "resume_path":     self.resume_path,
            "db_path":         self.db_path,
        }
