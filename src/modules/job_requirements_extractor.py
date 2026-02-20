# src/modules/job_requirements_extractor.py
"""
Job Requirements Extractor

Extracts structured information from a job description:
  - Job title
  - Company and location
  - Required hard skills
  - Nice-to-have skills
  - Required years of experience
  - Required education
"""

from typing import List, Dict, Tuple, Optional
import re

from src.modules.skills_extractor import SkillsExtractor


class JobRequirementsExtractor:
    """
    Extracts requirements from a job description text.

    Uses:
    - Heuristic section splitting (Requirements vs Nice to Have)
    - Your existing SkillsExtractor for skill detection
    - Regex patterns for experience years and education

    Usage:
        extractor = JobRequirementsExtractor()
        result = extractor.extract(job_text)
    """

    # Section heading keywords (lowercase)
    REQUIRED_HEADINGS = [
        "requirements", "required", "qualifications", "must have",
        "must-have", "basic qualifications", "minimum qualifications",
        "what you need", "what we need", "you will need",
        "technical requirements", "skills required"
    ]

    PREFERRED_HEADINGS = [
        "preferred", "nice to have", "nice-to-have", "bonus",
        "plus", "preferred qualifications", "additional qualifications",
        "good to have", "desirable", "advantageous",
        "what would be great", "extra credit"
    ]

    RESPONSIBILITY_HEADINGS = [
        "responsibilities", "what you will do", "what you'll do",
        "duties", "your role", "about the role", "the role",
        "key responsibilities", "day to day", "day-to-day"
    ]

    # Degree keywords for education extraction
    DEGREE_KEYWORDS = [
        "bachelor", "master", "phd", "ph.d", "doctorate",
        "bs", "ms", "msc", "bsc", "b.s", "m.s",
        "b.tech", "m.tech", "degree", "diploma"
    ]

    # Common job title words for validation
    TITLE_KEYWORDS = [
        "engineer", "developer", "scientist", "analyst", "manager",
        "designer", "architect", "lead", "director", "specialist",
        "consultant", "coordinator", "administrator", "intern",
        "associate", "senior", "junior", "staff", "principal"
    ]

    def __init__(self):
        """Initialize with skills extractor."""
        self.skills_extractor = SkillsExtractor()


    def extract(self, text: str) -> dict:
        """
        Extract all requirements from a job description.

        Args:
            text: Raw job description text

        Returns:
            Dictionary with keys:
              - title: str
              - company: str
              - location: str
              - required_hard_skills: List[str]
              - nice_to_have_skills: List[str]
              - required_experience_years: float
              - required_education: str
        """
        if not text or not text.strip():
            return self._empty_result()

        # Step 1: Split into sections
        sections = self._split_into_sections(text)

        # Step 2: Extract title, company, location from header
        title, company, location = self._extract_header_info(text)

        # Step 3: Extract skills from sections
        required_hard_skills = self._extract_skills_from_section(
            sections.get("required", "")
        )
        nice_to_have_skills = self._extract_skills_from_section(
            sections.get("preferred", "")
        )

        # Deduplicate nice-to-have against required
        nice_to_have_skills = [
            s for s in nice_to_have_skills
            if s not in required_hard_skills
        ]

        # Step 4: Extract experience years from required section
        # (or full text if section not found)
        search_text = sections.get("required", text)
        required_experience_years = self._extract_experience_years(search_text)

        # Step 5: Extract education
        required_education = self._extract_education_requirement(search_text)

        return {
            "title": title,
            "company": company,
            "location": location,
            "required_hard_skills": required_hard_skills,
            "nice_to_have_skills": nice_to_have_skills,
            "required_experience_years": required_experience_years,
            "required_education": required_education,
        }

    def _empty_result(self) -> dict:
        """Return empty result for empty input."""
        return {
            "title": "",
            "company": "",
            "location": "",
            "required_hard_skills": [],
            "nice_to_have_skills": [],
            "required_experience_years": 0.0,
            "required_education": "",
        }

    def _split_into_sections(self, text: str) -> Dict[str, str]:
        """
        Split job description into named sections.

        Returns a dict with keys:
          - "required": text under requirements heading
          - "preferred": text under nice-to-have heading
          - "responsibilities": text under responsibilities heading
          - "other": everything else
        """
        lines = text.splitlines()
        sections = {
            "required": [],
            "preferred": [],
            "responsibilities": [],
            "other": []
        }

        current_section = "other"

        for line in lines:
            stripped = line.strip()
            stripped_lower = stripped.lower()

            # Check if this line is a section heading
            is_required = any(
                h in stripped_lower
                for h in self.REQUIRED_HEADINGS
            )
            is_preferred = any(
                h in stripped_lower
                for h in self.PREFERRED_HEADINGS
            )
            is_responsibility = any(
                h in stripped_lower
                for h in self.RESPONSIBILITY_HEADINGS
            )

            # A heading line: short, likely no punctuation at end
            is_heading = (
                len(stripped) < 60 and
                (is_required or is_preferred or is_responsibility) and
                not stripped.endswith(",")
            )

            if is_heading:
                if is_required:
                    current_section = "required"
                elif is_preferred:
                    current_section = "preferred"
                elif is_responsibility:
                    current_section = "responsibilities"
                # Don't add the heading line itself
                continue

            sections[current_section].append(stripped)

        return {
            key: "\n".join(lines_list)
            for key, lines_list in sections.items()
        }

    def _extract_header_info(self, text: str) -> Tuple[str, str, str]:
        """
        Extract job title, company, and location from the top of the JD.

        Strategy:
        - First meaningful line → title (if it looks like a job title)
        - Second line → try to parse company | location or company, location

        Returns:
            (title, company, location)
        """
        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        if not lines:
            return "", "", ""

        title = ""
        company = ""
        location = ""

        # Line 1: Title
        first_line = lines[0]
        if self._looks_like_title(first_line):
            title = first_line

        # Line 2: Company | Location
        if len(lines) >= 2:
            second_line = lines[1]
            # Try splitting on | or ,
            for sep in ["|", "·", "—", "-", ","]:
                if sep in second_line:
                    parts = [p.strip() for p in second_line.split(sep)]
                    if len(parts) >= 2:
                        company = parts[0]
                        location = parts[1]
                        break
            else:
                # No separator → treat whole line as company
                company = second_line

        # Fallback: search for explicit labels anywhere in first 10 lines
        for line in lines[:10]:
            line_lower = line.lower()
            if "location:" in line_lower:
                location = line.split(":", 1)[-1].strip()
            if "company:" in line_lower:
                company = line.split(":", 1)[-1].strip()

        return title, company, location

    def _looks_like_title(self, text: str) -> bool:
        """
        Check if a text string looks like a job title.

        Criteria:
        - Length: 3-80 characters
        - Contains at least one known title keyword
        - Does not look like a heading (e.g., "About us", "Overview")
        """
        if not text or len(text) < 3 or len(text) > 80:
            return False

        text_lower = text.lower()
        has_title_word = any(kw in text_lower for kw in self.TITLE_KEYWORDS)
        return has_title_word

    def _extract_skills_from_section(self, section_text: str) -> List[str]:
        """
        Extract hard skills from a specific section of the JD.

        Reuses your existing SkillsExtractor (Module 1).

        Args:
            section_text: Text content of one JD section

        Returns:
            List of detected skill names
        """
        if not section_text or not section_text.strip():
            return []

        result = self.skills_extractor.extract_skills(section_text)

        if result is None:
            return []

        # Get hard skills from the result
        hard_skills = getattr(result, "hard_skills", [])
        return hard_skills if hard_skills else []

    def _extract_experience_years(self, text: str) -> float:
        """
        Extract minimum required experience years from text.

        Patterns:
        - "5+ years of experience"
        - "minimum 3 years"
        - "at least 7 years"
        - "2-4 years" → takes lower bound (2)
        - "3 to 5 years" → takes lower bound (3)

        Returns:
            Float years (0.0 if not found)
        """
        if not text:
            return 0.0

        patterns = [
            # "at least X years" / "minimum X years"
            r'(?:at\s+least|minimum|min\.?)\s+(\d+\.?\d*)\s*\+?\s*years?',
            # "X+ years of experience"
            r'(\d+\.?\d*)\s*\+\s*years?\s+of\s+(?:\w+\s+)?experience',
            # "X years of experience"
            r'(\d+\.?\d*)\s+years?\s+of\s+(?:\w+\s+)?experience',
            # "X-Y years" → take X (lower bound)
            r'(\d+)\s*[-–]\s*\d+\s*years?',
            # "X to Y years" → take X (lower bound)
            r'(\d+)\s+to\s+\d+\s*years?',
            # "X years working / in"
            r'(\d+\.?\d*)\s*\+?\s*years?\s+(?:working|in|as)',
        ]

        found = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = float(match)
                    if 0 < years <= 30:  # Sanity check
                        found.append(years)
                except ValueError:
                    continue

        # Return minimum found (most lenient / entry-level requirement)
        return min(found) if found else 0.0

    def _extract_education_requirement(self, text: str) -> str:
        """
        Extract education requirements from job description.

        Looks for lines containing degree keywords and returns
        the most informative one.

        Args:
            text: Requirements section text (or full JD)

        Returns:
            Education requirement string, or empty string
        """
        if not text:
            return ""

        lines = text.splitlines()
        education_lines = []

        for line in lines:
            line_stripped = line.strip()
            line_lower = line_stripped.lower()

            # Check if line contains any degree keyword
            has_degree = any(kw in line_lower for kw in self.DEGREE_KEYWORDS)

            if has_degree and len(line_stripped) > 5:
                # Clean up bullet points and leading symbols
                cleaned = re.sub(r'^[\-\*\•\–\·\s]+', '', line_stripped)
                if cleaned:
                    education_lines.append(cleaned)

        if not education_lines:
            return ""

        # Return the most informative line
        # (longest line tends to be most descriptive)
        best = max(education_lines, key=len)
        return best
