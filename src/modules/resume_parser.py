# src/modules/resume_parser.py
"""
Resume Parser core structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict
import numpy as np


@dataclass
class ResumeProfile:
    """
    Core data structure representing a parsed resume.

    Fields:
    - version: profile version or template id
    - hard_skills: list of technical skills
    - soft_skills: list of soft skills
    - education: formatted education string
    - total_experience_years: total years of professional experience
    - job_history: list of job history strings
    - resume_embedding: vector embedding representing full profile
    - skills_embeddings: mapping skill -> embedding vector
    - raw_text: raw or summary text used for embedding
    - created_at: timestamp when profile was created
    """
    version: str
    hard_skills: List[str]
    soft_skills: List[str]
    education: str
    total_experience_years: float
    job_history: List[str]
    resume_embedding: np.ndarray
    skills_embeddings: Dict[str, np.ndarray]
    raw_text: str
    created_at: datetime
