# src/modules/skills_extractor.py
"""
Skills Extraction Module
Uses the skills taxonomy and NLP to extract hard and soft skills from resume text.
"""

from typing import List, Tuple, Set
import re

import spacy

from src.modules.skills_taxonomy import SkillsTaxonomy


class SkillsExtractor:
    """
    Extract hard and soft skills from resume text.

    Combines:
    - Taxonomy-based exact/phrase matching
    - Optional spaCy-based enhancement (entities and noun chunks)
    """

    def __init__(self, use_nlp: bool = True):
        """
        Initialize the extractor.

        Args:
            use_nlp: If True, use spaCy to enhance extraction.
        """
        self.taxonomy = SkillsTaxonomy()
        self.use_nlp = use_nlp

        self._nlp = None
        if self.use_nlp:
            self._load_spacy_model()

    def _load_spacy_model(self):
        """Load spaCy English model lazily."""
        if self._nlp is not None:
            return

        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            ) from e


    def extract_skills(self, text: str) -> Tuple[List[str], List[str]]:
        """
        Extract hard and soft skills from resume text.

        Steps:
        1. Taxonomy-based extraction (primary).
        2. Optional NLP-based enhancement (secondary).
        3. Merge, normalize, and deduplicate.

        Args:
            text: Cleaned resume text.

        Returns:
            (hard_skills, soft_skills) - sorted, unique lists.
        """
        if not text or not text.strip():
            return [], []

        # Step 1: Taxonomy-based extraction
        hard_tax, soft_tax = self.taxonomy.extract_skills_from_text(text)

        hard_set: Set[str] = set(hard_tax)
        soft_set: Set[str] = set(soft_tax)

        # Step 2: Optional spaCy-based enhancement
        if self.use_nlp:
            doc = self._nlp(text)

            hard_nlp = self._extract_nlp_hard_skills(doc)
            soft_nlp = self._extract_nlp_soft_skills(doc)

            hard_set.update(hard_nlp)
            soft_set.update(soft_nlp)

        # Step 3: Final cleanup and sorting
        hard_final = self._postprocess_skills(hard_set, hard=True)
        soft_final = self._postprocess_skills(soft_set, hard=False)

        return hard_final, soft_final

    def _postprocess_skills(self, skills: Set[str], hard: bool = True) -> List[str]:
        """
        Final filtering and sorting of skills.

        - Remove empty/whitespace-only strings.
        - Remove 1-char noise (except explicit hard skills like 'c').
        - Normalize via taxonomy.
        - Keep only known skills.

        Args:
            skills: Set of candidate skills.
            hard: Whether these are hard skills.

        Returns:
            Sorted list of cleaned skills.
        """
        cleaned: Set[str] = set()

        for skill in skills:
            if not skill:
                continue

            s = skill.strip()
            if not s:
                continue

            # Filter out most 1-char tokens
            if len(s) == 1:
                if hard:
                    # Only keep if taxonomy explicitly knows this 1-char skill
                    if s not in self.taxonomy.all_hard_skills:
                        continue
                else:
                    # 1-char soft skills make no sense
                    continue

            # Normalize via taxonomy
            normalized = self.taxonomy.normalize_skill(s)

            if hard:
                if normalized in self.taxonomy.all_hard_skills:
                    cleaned.add(normalized)
            else:
                if normalized in self.taxonomy.soft_skills:
                    cleaned.add(normalized)

        return sorted(cleaned)

    def _extract_nlp_hard_skills(self, doc) -> List[str]:
        """
        Use spaCy entities and noun chunks to find additional hard skills.

        Strategy:
        - Look at ORG, PRODUCT, FAC entities (often tools/technologies).
        - Look at noun chunks that contain specific skill phrases.
        - Normalize and keep only skills present in taxonomy.
        """
        candidates: Set[str] = set()

        # 1. Entity-based candidates
        for ent in doc.ents:
            if ent.label_ in {"ORG", "PRODUCT", "FAC"}:
                phrase = ent.text.strip()
                # Normalize whole phrase
                norm_phrase = self.taxonomy.normalize_skill(phrase)
                if norm_phrase in self.taxonomy.all_hard_skills:
                    candidates.add(norm_phrase)

                # Also split phrase into tokens - e.g. "AWS Cloud"
                for token in phrase.split():
                    t_norm = self.taxonomy.normalize_skill(token)
                    if t_norm in self.taxonomy.all_hard_skills:
                        candidates.add(t_norm)

        # 2. Noun chunk-based candidates for known phrases
        anchor_keywords = [
            "machine learning",
            "deep learning",
            "neural network",
            "computer vision",
            "natural language processing",
            "nlp",
            "rest api",
            "microservice",
        ]

        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.strip().lower()

            for anchor in anchor_keywords:
                if anchor in chunk_text:
                    norm = self.taxonomy.normalize_skill(anchor)
                    if norm in self.taxonomy.all_hard_skills:
                        candidates.add(norm)

        return list(candidates)

    def _extract_nlp_soft_skills(self, doc) -> List[str]:
        """
        Reinforce soft skills via simple pattern-based matching.

        Currently:
        - Reuse taxonomy soft skills and search them in the text.
        - This overlaps with taxonomy extraction but gives a hook
          for more complex patterns in the future.
        """
        candidates: Set[str] = set()
        text_lower = doc.text.lower()

        for soft_skill in self.taxonomy.soft_skills:
            pattern = r'\b' + re.escape(soft_skill) + r'\b'
            if re.search(pattern, text_lower):
                candidates.add(soft_skill)

        return list(candidates)

