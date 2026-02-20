# src/modules/resume_orchestrator.py
"""
Resume Orchestrator

High-level pipeline that:
- Takes a resume (PDF or text)
- Runs all extractors
- Creates embeddings
- Builds a ResumeProfile
- Optionally saves it to JSON
"""

from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np

from src.modules.resume_parser import ResumeProfile
from src.modules.pdf_extractor import PDFExtractor           # your PDF module name
from src.modules.skills_extractor import SkillsExtractor         # your skills module
from src.modules.education_extractor import EducationExtractor
from src.modules.experience_extractor import ExperienceExtractor
from src.modules.job_history_extractor import JobHistoryExtractor
from src.modules.embeddings_creator import EmbeddingsCreator
from src.modules.profile_serializer import ProfileSerializer


class ResumeOrchestrator:
    """
    Orchestrates the end-to-end resume parsing pipeline.
    """

    def __init__(self):
        # Core components
        self.pdf_extractor = PDFExtractor()
        self.skills_extractor = SkillsExtractor()
        self.education_extractor = EducationExtractor(use_nlp=True)
        self.experience_extractor = ExperienceExtractor()
        self.job_history_extractor = JobHistoryExtractor(use_nlp=True)
        self.embeddings_creator = EmbeddingsCreator()
        self.serializer = ProfileSerializer()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_pdf(self, pdf_path: str, profile_version: str) -> ResumeProfile:
        """
        End-to-end processing of a PDF resume.

        Steps:
        1. Extract text from PDF.
        2. Extract skills, education, experience years, job history.
        3. Build summary and embeddings.
        4. Build ResumeProfile.

        Args:
            pdf_path: Path to the PDF resume.
            profile_version: Label/version name for this profile.

        Returns:
            ResumeProfile instance.
        """
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # 1. Text extraction
        text = self._extract_text_from_pdf(pdf_path_obj)

        # 2. Info extraction
        hard_skills, soft_skills = self._extract_skills(text)
        education = self.education_extractor.extract_education(text)
        years_exp = self.experience_extractor.extract_years_of_experience(text)
        job_entries = self.job_history_extractor.extract_job_history(text)
        job_history_strs = [entry.format() for entry in job_entries]

        # 3. Build summary text for embedding
        summary_text = self.embeddings_creator.build_resume_summary_text(
            hard_skills=hard_skills,
            soft_skills=soft_skills,
            education=education,
            experience_years=years_exp,
            job_history=job_history_strs,
        )

        # 4. Create embeddings
        resume_embedding = self.embeddings_creator.create_text_embedding(summary_text)
        skills_embeddings = self.embeddings_creator.create_skills_embeddings(hard_skills)

        # 5. Build ResumeProfile
        profile = ResumeProfile(
            version=profile_version,
            hard_skills=hard_skills,
            soft_skills=soft_skills,
            education=education,
            resume_embedding=resume_embedding,
            skills_embeddings=skills_embeddings,
            created_at=datetime.now(),
        )

        return profile

    def process_text(self, text: str, profile_version: str) -> ResumeProfile:
        """
        Same as process_pdf, but starting from raw text instead of a PDF file.
        """
        if not text or not text.strip():
            raise ValueError("Empty resume text provided")

        hard_skills, soft_skills = self._extract_skills(text)
        education = self.education_extractor.extract_education(text)
        years_exp = self.experience_extractor.extract_years_of_experience(text)
        job_entries = self.job_history_extractor.extract_job_history(text)
        job_history_strs = [entry.format() for entry in job_entries]

        summary_text = self.embeddings_creator.build_resume_summary_text(
            hard_skills=hard_skills,
            soft_skills=soft_skills,
            education=education,
            experience_years=years_exp,
            job_history=job_history_strs,
        )

        resume_embedding = self.embeddings_creator.create_text_embedding(summary_text)
        skills_embeddings = self.embeddings_creator.create_skills_embeddings(hard_skills)

        profile = ResumeProfile(
            version=profile_version,
            hard_skills=hard_skills,
            soft_skills=soft_skills,
            education=education,
            resume_embedding=resume_embedding,
            skills_embeddings=skills_embeddings,
            created_at=datetime.now(),
        )

        return profile

    def save_profile_json(self, profile: ResumeProfile, output_path: str) -> str:
        """
        Save a ResumeProfile to JSON using ProfileSerializer.
        """
        return self.serializer.save_profile(profile, output_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Wrapper around your PDF extraction logic.
        """
        # Adjust according to your PdfTextExtractor API
        # Example:
        # return self.pdf_extractor.extract_text(str(pdf_path))
        return self.pdf_extractor.extract_text(str(pdf_path))

    def _extract_skills(self, text: str):
        """
        Wrapper around your SkillsExtractor to return (hard_skills, soft_skills).
        """
        # Adjust according to your SkillsExtractor API
        # Example:
        # result = self.skills_extractor.extract_skills(text)
        # return result.hard_skills, result.soft_skills
        result = self.skills_extractor.extract_skills(text)
        hard_skills = getattr(result, "hard_skills", []) if result is not None else []
        soft_skills = getattr(result, "soft_skills", []) if result is not None else []
        return hard_skills, soft_skills
