# src/modules/profile_serializer.py
"""
Profile Serialization Module
Saves and loads ResumeProfile objects to/from JSON files,
handling numpy arrays and datetime objects.
"""

import json
from pathlib import Path
from datetime import datetime

import numpy as np

from src.modules.resume_parser import ResumeProfile


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy arrays and datetime objects.
    """

    def default(self, obj):
        # numpy arrays
        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "data": obj.tolist(),
                "dtype": str(obj.dtype),
                "shape": list(obj.shape),
            }
        # numpy scalar types
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        # datetime objects
        if isinstance(obj, datetime):
            return {
                "__type__": "datetime",
                "value": obj.isoformat(),
            }
        # Fallback to default
        return super().default(obj)


def numpy_decoder_hook(obj: dict):
    """
    Custom JSON decoder hook to restore numpy arrays and datetime objects.
    """
    if "__type__" in obj:
        t = obj["__type__"]
        if t == "ndarray":
            return np.array(obj["data"], dtype=obj["dtype"]).reshape(obj["shape"])
        if t == "datetime":
            return datetime.fromisoformat(obj["value"])
    return obj


class ProfileSerializer:
    """
    Save and load ResumeProfile objects to/from JSON files.
    """

    def save_profile(self, profile: ResumeProfile, filepath: str) -> str:
        """
        Save a ResumeProfile to a JSON file.

        Args:
            profile: ResumeProfile instance
            filepath: Path to output JSON

        Returns:
            Absolute path to saved file.
        """
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        profile_dict = self._profile_to_dict(profile)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                profile_dict,
                f,
                cls=NumpyEncoder,
                indent=2,
                ensure_ascii=False,
            )

        return str(output_path.resolve())

    def load_profile(self, filepath: str) -> ResumeProfile:
        """
        Load a ResumeProfile from a JSON file.

        Args:
            filepath: Path to JSON file

        Returns:
            ResumeProfile instance.

        Raises:
            FileNotFoundError if file does not exist.
        """
        input_path = Path(filepath)
        if not input_path.exists():
            raise FileNotFoundError(f"Profile not found: {filepath}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f, object_hook=numpy_decoder_hook)

        return self._dict_to_profile(data)

    def _profile_to_dict(self, profile: ResumeProfile) -> dict:
        """
        Convert ResumeProfile to a plain dict.
        """
        return {
            "version": profile.version,
            "hard_skills": profile.hard_skills,
            "soft_skills": profile.soft_skills,
            "education": profile.education,
            "resume_embedding": profile.resume_embedding,
            "skills_embeddings": profile.skills_embeddings,
            "created_at": profile.created_at,
        }

    def _dict_to_profile(self, data: dict) -> ResumeProfile:
        """
        Convert dict back to ResumeProfile.
        """
        return ResumeProfile(
            version=data.get("version", "unknown"),
            hard_skills=data.get("hard_skills", []),
            soft_skills=data.get("soft_skills", []),
            education=data.get("education", ""),
            resume_embedding=data.get("resume_embedding", np.zeros(384)),
            skills_embeddings=data.get("skills_embeddings", {}),
            created_at=data.get("created_at", datetime.now()),
        )

    def profile_exists(self, filepath: str) -> bool:
        """
        Check if a profile JSON file exists.
        """
        return Path(filepath).exists()
