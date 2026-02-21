# src/modules/embeddings_creator.py
"""
Embeddings Creation Module
Converts resume text and skills into numerical vector embeddings
for semantic similarity matching.
"""

from typing import List, Dict, Optional
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingsCreator:
    """
    Create embeddings for resume text and skills using Sentence Transformers.
    
    Uses all-MiniLM-L6-v2 model:
    - Output dimension: 384
    - Speed: ~14,000 sentences/second (CPU)
    - Good balance of speed and accuracy
    
    Usage:
        creator = EmbeddingsCreator()
        embedding = creator.create_text_embedding("Python developer")
        print(embedding.shape)  # (384,)
    """
    
    # Model name (class-level constant)
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize the embeddings creator.
        
        Args:
            model_name: Sentence Transformer model name.
                       Defaults to 'all-MiniLM-L6-v2'.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        
        # Lazy loading: model loads on first use
        # This avoids slow startup if embeddings aren't needed
    
    def _load_model(self):
        """Load the Sentence Transformer model lazily."""
        if self._model is not None:
            return
        
        print(f"Loading embedding model: {self.model_name}...")
        self._model = SentenceTransformer(self.model_name)
        print(f"✓ Model loaded (dimension: {self.EMBEDDING_DIM})")

    def create_text_embedding(self, text: str) -> np.ndarray:
        """
        Create embedding for a single text string.
        
        Args:
            text: Any text (resume summary, job description, etc.)
            
        Returns:
            numpy array of shape (384,)
            
        Example:
            >>> creator = EmbeddingsCreator()
            >>> emb = creator.create_text_embedding("Python developer")
            >>> print(emb.shape)
            (384,)
        """
        self._load_model()
        
        if not text or not text.strip():
            # Return zero vector for empty text
            return np.zeros(self.EMBEDDING_DIM)
        
        # Encode single text
        embedding = self._model.encode(text, show_progress_bar=False)
        
        return embedding.astype(np.float32)

    def create_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for multiple texts at once (much faster than one-by-one).
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (n_texts, 384)
            
        Example:
            >>> texts = ["Python developer", "Java engineer", "Data scientist"]
            >>> embs = creator.create_batch_embeddings(texts)
            >>> print(embs.shape)
            (3, 384)
        """
        self._load_model()
        
        if not texts:
            return np.zeros((0, self.EMBEDDING_DIM))
        
        # Replace empty strings with placeholder
        cleaned = [t if t and t.strip() else "empty" for t in texts]
        
        # Batch encode (much faster than one-by-one)
        embeddings = self._model.encode(
            cleaned,
            show_progress_bar=False,
            batch_size=32
        )
        
        return embeddings.astype(np.float32)

    def create_skills_embeddings(self, skills: List[str]) -> Dict[str, np.ndarray]:
        """
        Create embeddings for a list of skills.
        
        Returns a dictionary mapping each skill name to its embedding vector.
        
        Args:
            skills: List of skill names (e.g., ["python", "tensorflow", "aws"])
            
        Returns:
            Dictionary: {skill_name: embedding_vector}
            
        Example:
            >>> skills_embs = creator.create_skills_embeddings(["python", "aws"])
            >>> print(skills_embs["python"].shape)
            (384,)
        """
        if not skills:
            return {}
        
        # Remove duplicates while preserving order
        unique_skills = list(dict.fromkeys(skills))
        
        # Batch encode all skills at once
        embeddings = self.create_batch_embeddings(unique_skills)
        
        # Build dictionary
        skills_dict = {}
        for skill, embedding in zip(unique_skills, embeddings):
            skills_dict[skill] = embedding
        
        return skills_dict

    def compute_similarity(self, embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding_a: First embedding vector (384,)
            embedding_b: Second embedding vector (384,)
            
        Returns:
            Similarity score between 0.0 and 1.0
            
        Example:
            >>> sim = creator.compute_similarity(resume_emb, job_emb)
            >>> print(f"Match: {sim:.2f}")
            0.78
        """
        # Reshape to 2D for sklearn (expects matrix input)
        a = embedding_a.reshape(1, -1)
        b = embedding_b.reshape(1, -1)
        
        similarity = cosine_similarity(a, b)[0][0]
        
        # Clamp to [0, 1] range (cosine can technically be -1 to 1)
        return float(max(0.0, min(1.0, similarity)))
    
    def compute_batch_similarity(
        self,
        embedding: np.ndarray,
        embeddings_list: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between one embedding and many others.
        
        Useful for comparing one resume against multiple job descriptions.
        
        Args:
            embedding: Single embedding (384,)
            embeddings_list: Multiple embeddings (n, 384)
            
        Returns:
            Array of similarity scores (n,)
            
        Example:
            >>> resume_emb = creator.create_text_embedding("Python dev")
            >>> job_embs = creator.create_batch_embeddings(["Job A", "Job B", "Job C"])
            >>> scores = creator.compute_batch_similarity(resume_emb, job_embs)
            >>> print(scores)
            [0.72, 0.45, 0.88]
        """
        query = embedding.reshape(1, -1)
        
        similarities = cosine_similarity(query, embeddings_list)[0]
        
        # Clamp to [0, 1]
        return np.clip(similarities, 0.0, 1.0)

    def build_resume_summary_text(
        self,
        hard_skills: List[str],
        soft_skills: List[str],
        education: str,
        experience_years: float,
        job_history: List[str]
    ) -> str:
        """
        Build a text summary of the resume for embedding.
        
        Combines all extracted information into a single paragraph
        that captures the candidate's profile.
        
        Args:
            hard_skills: List of technical skills
            soft_skills: List of soft skills
            education: Education string
            experience_years: Years of experience
            job_history: List of job history strings
            
        Returns:
            Summary text for embedding
            
        Example:
            >>> summary = creator.build_resume_summary_text(
            ...     hard_skills=["python", "aws"],
            ...     soft_skills=["leadership"],
            ...     education="Master in CS | Stanford",
            ...     experience_years=5.0,
            ...     job_history=["Data Scientist at Google"]
            ... )
            >>> print(summary)
            "Professional with 5.0 years of experience. Technical skills: python, aws.
             Soft skills: leadership. Education: Master in CS | Stanford.
             Experience: Data Scientist at Google."
        """
        parts = []
        
        # Experience
        if experience_years > 0:
            parts.append(f"Professional with {experience_years:.1f} years of experience.")
        
        # Hard skills
        if hard_skills:
            skills_str = ", ".join(hard_skills[:20])  # Limit to top 20
            parts.append(f"Technical skills: {skills_str}.")
        
        # Soft skills
        if soft_skills:
            soft_str = ", ".join(soft_skills[:10])  # Limit to top 10
            parts.append(f"Soft skills: {soft_str}.")
        
        # Education
        if education:
            parts.append(f"Education: {education}.")
        
        # Job history
        if job_history:
            jobs_str = "; ".join(job_history[:5])  # Limit to top 5
            parts.append(f"Experience: {jobs_str}.")
        
        return " ".join(parts) if parts else "No profile information available."

    def get_embedding_info(self) -> dict:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary with model information
        """
        self._load_model()
        
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.EMBEDDING_DIM,
            "max_sequence_length": self._model.max_seq_length,
        }

    #The next two methods are for job descriptions, which have a slightly different structure than resumes.
    
    def build_job_summary_text(
        self,
        title: str,
        required_hard_skills: List[str],
        nice_to_have_skills: List[str],
        required_experience_years: float,
        required_education: str,
    ) -> str:
        """
        Build a text summary of a job description for embedding.

        Combines all job requirements into a single structured paragraph
        that captures the role's semantic meaning.

        Args:
            title: Job title (e.g. "Senior Data Scientist")
            required_hard_skills: Must-have technical skills
            nice_to_have_skills: Preferred/bonus skills
            required_experience_years: Minimum years of experience
            required_education: Education requirement string

        Returns:
            Summary text ready for embedding
        """
        parts = []

        # Title + experience
        if title:
            if required_experience_years > 0:
                parts.append(
                    f"{title} role requiring {required_experience_years:.1f}"
                    f" years of experience."
                )
            else:
                parts.append(f"{title} role.")

        # Required skills
        if required_hard_skills:
            skills_str = ", ".join(required_hard_skills[:20])
            parts.append(f"Required technical skills: {skills_str}.")

        # Nice-to-have skills
        if nice_to_have_skills:
            preferred_str = ", ".join(nice_to_have_skills[:10])
            parts.append(f"Preferred skills: {preferred_str}.")

        # Education
        if required_education:
            parts.append(f"Education requirement: {required_education}.")

        return " ".join(parts) if parts else "No job information available."

    def create_job_embeddings(
        self,
        title: str,
        required_hard_skills: List[str],
        nice_to_have_skills: List[str],
        required_experience_years: float,
        required_education: str,
    ) -> dict:
        """
        Create all embeddings needed for a JobProfile in one call.

        Returns a dictionary with:
          - "job_embedding":    np.ndarray shape (384,)
          - "skills_embeddings": Dict[str, np.ndarray] per-skill vectors
          - "summary_text":     str used for embedding (for debugging)

        Args:
            title: Job title
            required_hard_skills: Must-have skills
            nice_to_have_skills: Preferred skills
            required_experience_years: Minimum experience years
            required_education: Education requirement string

        Returns:
            Dictionary with job_embedding, skills_embeddings, summary_text
        """
        # Build the summary text
        summary_text = self.build_job_summary_text(
            title=title,
            required_hard_skills=required_hard_skills,
            nice_to_have_skills=nice_to_have_skills,
            required_experience_years=required_experience_years,
            required_education=required_education,
        )

        # Create job-level embedding from summary
        job_embedding = self.create_text_embedding(summary_text)

        # Deduplicate and embed all skills (required + nice-to-have)
        all_skills = list(dict.fromkeys(
            required_hard_skills + nice_to_have_skills
        ))
        skills_embeddings = self.create_skills_embeddings(all_skills)

        return {
            "job_embedding": job_embedding,
            "skills_embeddings": skills_embeddings,
            "summary_text": summary_text,
        }
