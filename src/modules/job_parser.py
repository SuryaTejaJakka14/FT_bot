# src/modules/job_parser.py
"""
Job Parser core structures.
Defines the JobProfile dataclass used to represent a parsed job description.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict
import numpy as np


@dataclass
class JobProfile:
    """
    Structured representation of a job description.

    Fields:
        version: Profile schema/version label.
        title: Job title (e.g., 'Senior Data Scientist').
        company: Company name, if known (may be empty).
        location: Job location, if known (may be empty).
        required_hard_skills: List of required technical skills.
        nice_to_have_skills: List of preferred / bonus skills.
        required_experience_years: Minimum years of experience required.
        required_education: Text describing education requirements.
        job_embedding: Embedding vector representing the job as a whole.
        skills_embeddings: Mapping skill -> embedding vector.
        created_at: Timestamp when this JobProfile was created.
    """
    version: str
    title: str
    company: str
    location: str
    required_hard_skills: List[str]
    nice_to_have_skills: List[str]
    required_experience_years: float
    required_education: str
    job_embedding: np.ndarray
    skills_embeddings: Dict[str, np.ndarray]
    created_at: datetime
