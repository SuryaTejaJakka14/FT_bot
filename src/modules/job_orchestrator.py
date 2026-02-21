# src/modules/job_orchestrator.py
"""
JobOrchestrator: Coordinates all Module 2 components to process
a raw job description into a fully populated JobProfile.

Pipeline:
    Raw JD text
        → JobTextNormalizer        (normalize + clean)
        → JobRequirementsExtractor (title, skills, experience, education)
        → EmbeddingsCreator        (job + skills embeddings)
        → JobProfile               (final dataclass output)
"""

import logging
import numpy as np
from datetime import datetime
from typing import Optional

from src.modules.job_parser import JobProfile
from src.modules.job_text_normaliser import JobTextNormalizer
from src.modules.job_requirements_extractor import JobRequirementsExtractor
from src.modules.embeddings_creator import EmbeddingsCreator

logger = logging.getLogger(__name__)


class JobOrchestrator:
    """
    Orchestrates the full job description processing pipeline.

    Coordinates:
      - JobTextNormalizer:        cleans and normalizes raw JD text
      - JobRequirementsExtractor: extracts structured requirements
      - EmbeddingsCreator:        creates job + skills embeddings

    Usage:
        orchestrator = JobOrchestrator()
        profile = orchestrator.process(jd_text)

        print(profile.title)                # "Senior Data Scientist"
        print(profile.required_hard_skills) # ["python", "tensorflow", ...]
        print(profile.job_embedding.shape)  # (384,)
    """

    def __init__(self, embeddings_creator: Optional[EmbeddingsCreator] = None):
        """
        Initialize all pipeline components.

        Args:
            embeddings_creator: Optional shared EmbeddingsCreator instance.
                                 Pass one in to avoid reloading the model
                                 when you already have it loaded elsewhere.
                                 If None, a new instance is created.
        """
        self.normalizer = JobTextNormalizer()
        self.extractor  = JobRequirementsExtractor()
        self.embedder   = embeddings_creator or EmbeddingsCreator()

        logger.info("JobOrchestrator initialized")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def process(self, jd_text: str) -> JobProfile:
        """
        Process raw job description text into a fully populated JobProfile.

        This is the single entry point for the entire Module 2 pipeline.

        Args:
            jd_text: Raw job description text (any format)

        Returns:
            JobProfile dataclass with all fields populated

        Example:
            >>> orchestrator = JobOrchestrator()
            >>> profile = orchestrator.process(jd_text)
            >>> print(profile.title)
            'Senior Data Scientist'
            >>> print(profile.job_embedding.shape)
            (384,)
        """
        if not jd_text or not jd_text.strip():
            logger.warning("Empty JD text received — returning empty JobProfile")
            return JobProfile(
                version="1.0",
                title="",
                company="",
                location="",
                required_hard_skills=[],
                nice_to_have_skills=[],
                required_experience_years=0.0,
                required_education="",
                job_embedding=np.zeros(384),
                skills_embeddings={},
                created_at=datetime.now(),
            )

        logger.info("Starting job description processing pipeline")

        # Step 1: Normalize
        normalized_text = self._normalize(jd_text)

        # Step 2: Extract requirements
        requirements = self._extract_requirements(normalized_text)

        # Step 3: Create embeddings
        embeddings = self._create_embeddings(requirements)

        # Step 4: Assemble final profile
        profile = self._build_profile(requirements, embeddings)

        logger.info(
            f"Pipeline complete — title='{profile.title}', "
            f"required_skills={len(profile.required_hard_skills)}, "
            f"nice_to_have={len(profile.nice_to_have_skills)}"
        )

        return profile

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _normalize(self, jd_text: str) -> str:
        """
        Step 1: Clean and normalize raw JD text.

        Args:
            jd_text: Raw job description text

        Returns:
            Cleaned, normalized text
        """
        normalized = self.normalizer.normalize(jd_text)
        logger.debug(f"Normalized text length: {len(normalized)} chars")
        return normalized

    def _extract_requirements(self, normalized_text: str) -> dict:
        """
        Step 2: Extract all structured requirements from normalized text.

        Delegates to JobRequirementsExtractor and returns results
        as a plain dictionary for easy passing between steps.

        Args:
            normalized_text: Cleaned JD text from Step 1

        Returns:
            Dictionary with keys:
              title, company, location,
              required_hard_skills, nice_to_have_skills,
              required_experience_years, required_education
        """
        result = self.extractor.extract(normalized_text)

        logger.debug(
            f"Extracted — title='{result.get('title')}', "
            f"required={result.get('required_hard_skills', [])}"
        )

        return result

    def _create_embeddings(self, requirements: dict) -> dict:
        """
        Step 3: Create job-level and skill-level embeddings.

        Delegates to EmbeddingsCreator.create_job_embeddings().

        Args:
            requirements: Dictionary output from Step 2

        Returns:
            Dictionary with keys:
              job_embedding, skills_embeddings, summary_text
        """
        embeddings = self.embedder.create_job_embeddings(
            title=requirements.get("title", ""),
            required_hard_skills=requirements.get("required_hard_skills", []),
            nice_to_have_skills=requirements.get("nice_to_have_skills", []),
            required_experience_years=requirements.get("required_experience_years", 0.0),
            required_education=requirements.get("required_education", ""),
        )

        logger.debug(
            f"Embeddings created — "
            f"job_embedding shape: {embeddings['job_embedding'].shape}, "
            f"skills: {list(embeddings['skills_embeddings'].keys())}"
        )

        return embeddings

    def _build_profile(self, requirements: dict, embeddings: dict) -> JobProfile:
        """
        Step 4: Assemble final JobProfile dataclass from all extracted data.

        Args:
            requirements: Dictionary from Step 2 (extracted fields)
            embeddings:   Dictionary from Step 3 (vectors)

        Returns:
            Fully populated JobProfile dataclass
        """
        return JobProfile(
            version="1.0",
            title=requirements.get("title", ""),
            company=requirements.get("company", ""),
            location=requirements.get("location", ""),
            required_hard_skills=requirements.get("required_hard_skills", []),
            nice_to_have_skills=requirements.get("nice_to_have_skills", []),
            required_experience_years=requirements.get("required_experience_years", 0.0),
            required_education=requirements.get("required_education", ""),
            job_embedding=embeddings["job_embedding"],
            skills_embeddings=embeddings["skills_embeddings"],
            created_at=datetime.now(),
        )
