# src/modules/pipeline_scraper.py
"""
PipelineJobScraper: Adapter between the existing JobScraper
(job_scraper.py) and the Module 5 pipeline.

Responsibilities:
  1. Call existing JobScraper with PipelineConfig settings
  2. Convert JobPosting → ScrapedJob wrapping a JobProfile
  3. Extract required/nice-to-have skills from description text
  4. Return List[ScrapedJob] ready for MatchingEngine + JobStore

Does NOT modify job_scraper.py — it wraps it.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from datetime    import datetime
from typing      import List, Optional, Tuple

import numpy as np

from src.modules.job_parser      import JobProfile
from src.modules.job_scraper     import JobScraper, JobPosting
from src.modules.pipeline_config import PipelineConfig

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# SKILLS VOCABULARY  (for extraction from description text)
# ------------------------------------------------------------------

TECH_SKILLS = {
    # Languages
    "python", "java", "scala", "r", "sql", "javascript", "typescript",
    "c++", "c#", "go", "rust", "swift", "kotlin", "bash", "shell",
    # ML / AI
    "machine learning", "deep learning", "nlp", "computer vision",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost",
    "transformers", "llm", "reinforcement learning",
    # Data
    "pandas", "numpy", "spark", "hadoop", "airflow", "dbt", "kafka",
    "postgresql", "postgres", "mysql", "mongodb", "redis",
    "elasticsearch", "snowflake", "bigquery", "databricks",
    # Cloud & DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "ci/cd", "jenkins", "github actions", "ansible",
    # Tools
    "git", "linux", "tableau", "power bi", "excel", "jira",
    "mlflow", "wandb", "hugging face",
}

REQUIRED_SIGNALS = {
    "required", "must have", "must-have", "mandatory", "essential",
    "minimum", "necessary", "experience with", "proficiency in",
    "expertise in", "strong knowledge", "solid understanding",
}

NICE_TO_HAVE_SIGNALS = {
    "preferred", "nice to have", "nice-to-have", "bonus", "a plus",
    "an advantage", "desired", "ideally", "familiarity with",
    "exposure to", "knowledge of", "optional", "would be great",
}

EMBEDDING_DIM = 384


# ------------------------------------------------------------------
# OUTPUT WRAPPER
# ------------------------------------------------------------------

@dataclass
class ScrapedJob:
    """
    Wraps a JobProfile with scraper metadata needed by JobStore.

    Fields:
        profile:      JobProfile ready for MatchingEngine.match()
        job_id:       Stable dedup key (16-char hex)
        url:          Original job posting URL
        date_posted:  When the job was posted (ISO string)
        source:       Which site ("linkedin", "indeed", etc.)
        raw_title:    Original job title
        company:      Company name
        location:     Job location string
    """
    profile:     JobProfile
    job_id:      str
    url:         str
    date_posted: str
    source:      str
    raw_title:   str
    company:     str
    location:    str


# ------------------------------------------------------------------
# PIPELINE JOB SCRAPER
# ------------------------------------------------------------------

class PipelineJobScraper:
    """
    Adapter that connects the existing JobScraper to the Module 5 pipeline.

    Internally uses JobScraper (job_scraper.py) for HTTP scraping,
    then converts JobPosting → ScrapedJob with a JobProfile for
    MatchingEngine compatibility.

    Usage:
        scraper = PipelineJobScraper(config)
        jobs    = scraper.scrape()
        # jobs[0].profile is a JobProfile ready for MatchingEngine
        # jobs[0].job_id  is a dedup key for JobStore
    """

    def __init__(self, config: PipelineConfig, enable_selenium: bool = False):
        """
        Args:
            config:          PipelineConfig with search settings
            enable_selenium: Pass True to enable Selenium-based scrapers
                             (requires Chrome + chromedriver installed)
        """
        self.config  = config
        self._inner  = JobScraper(enable_selenium=enable_selenium)
        logger.info(
            f"PipelineJobScraper initialized: "
            f"role='{config.search_role}', "
            f"location='{config.location}'"
        )

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def scrape(self) -> List[ScrapedJob]:
        """
        Scrape jobs using config settings and return ScrapedJob list.

        Returns:
            List[ScrapedJob] ready for MatchingEngine + JobStore.
            Returns [] if scraping fails or returns no results.
        """
        logger.info(
            f"Scraping '{self.config.search_role}' on "
            f"{self.config.job_sites} ..."
        )

        # Use existing scraper's parallel method
        postings = self._inner.scrape_all_sources_parallel(
            queries  = [self.config.search_role],
            location = self.config.location,
        )

        if not postings:
            logger.warning("Scraper returned no results")
            return []

        logger.info(f"Raw postings received: {len(postings)}")

        jobs = [
            converted
            for posting in postings
            if (converted := self._convert(posting)) is not None
        ]

        logger.info(f"Successfully converted {len(jobs)}/{len(postings)}")
        return jobs

    def scrape_from_postings(
        self, postings: List[JobPosting]
    ) -> List[ScrapedJob]:
        """
        Convert an existing list of JobPosting objects to ScrapedJobs.

        Useful for testing or when postings are sourced externally.

        Args:
            postings: List[JobPosting] from existing JobScraper

        Returns:
            List[ScrapedJob]
        """
        jobs = [
            converted
            for posting in postings
            if (converted := self._convert(posting)) is not None
        ]
        logger.debug(
            f"Converted {len(jobs)}/{len(postings)} postings"
        )
        return jobs

    def close(self):
        """Close the underlying Selenium driver if open."""
        self._inner.close()

    # ------------------------------------------------------------------
    # PRIVATE: CONVERSION
    # ------------------------------------------------------------------

    def _convert(self, posting: JobPosting) -> Optional[ScrapedJob]:
        """Convert one JobPosting → ScrapedJob with JobProfile."""
        try:
            # Use existing requirements list if populated,
            # otherwise extract from description text
            if posting.requirements:
                required     = [r.lower().strip()
                                for r in posting.requirements if r.strip()]
                nice_to_have = []
            else:
                required, nice_to_have = self._extract_skills(
                    posting.description
                )

            # Fallback if no skills found at all
            if not required:
                required = [self.config.search_role.lower()]

            seed    = abs(hash(posting.title + posting.company)) % (2**31)
            profile = JobProfile(
                version                   = "1.0",
                title                     = posting.title,
                company                   = posting.company,
                location                  = posting.location,
                required_hard_skills      = required,
                nice_to_have_skills       = nice_to_have,
                required_experience_years = 0.0,
                required_education        = "",
                job_embedding             = self._unit_vec(seed),
                skills_embeddings         = {
                    s: self._unit_vec(seed + i + 1)
                    for i, s in enumerate(required + nice_to_have)
                },
                created_at                = datetime.now(),
            )

            job_id = self._make_job_id(
                posting.title, posting.company, posting.url
            )

            return ScrapedJob(
                profile     = profile,
                job_id      = job_id,
                url         = posting.url,
                date_posted = posting.created_at.strftime("%Y-%m-%d")
                              if posting.created_at else
                              datetime.now().strftime("%Y-%m-%d"),
                source      = posting.source,
                raw_title   = posting.title,
                company     = posting.company,
                location    = posting.location,
            )

        except Exception as e:
            logger.warning(
                f"Failed to convert posting "
                f"'{getattr(posting, 'title', '?')}': {e}"
            )
            return None

    # ------------------------------------------------------------------
    # PRIVATE: SKILL EXTRACTION
    # ------------------------------------------------------------------

    def _extract_skills(self, description: str) -> Tuple[List[str], List[str]]:
        """
        Extract required and nice-to-have skills from description text.

        Returns:
            (required_skills, nice_to_have_skills)
        """
        if not description:
            return [], []

        text      = description.lower()
        sentences = re.split(r'[.\n•\-]', text)

        required     = set()
        nice_to_have = set()

        for sentence in sentences:
            found = {s for s in TECH_SKILLS if s in sentence}
            if not found:
                continue

            if any(sig in sentence for sig in NICE_TO_HAVE_SIGNALS):
                nice_to_have.update(found)
            else:
                required.update(found)

        # Required takes priority over nice-to-have
        nice_to_have -= required

        return sorted(required), sorted(nice_to_have)

    # ------------------------------------------------------------------
    # PRIVATE: UTILITIES
    # ------------------------------------------------------------------

    def _make_job_id(self, title: str, company: str, url: str) -> str:
        """16-char dedup key from title+company+url."""
        raw = f"{title.lower()}|{company.lower()}|{url.lower()}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]

    def _unit_vec(self, seed: int) -> np.ndarray:
        """Reproducible random unit vector for placeholder embeddings."""
        rng = np.random.RandomState(seed % (2**31))
        v   = rng.randn(EMBEDDING_DIM)
        return (v / np.linalg.norm(v)).astype(np.float32)
