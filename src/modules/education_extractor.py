# src/modules/education_extractor.py
"""
Education Extraction Module
Extracts education information (degree, field, institution) from resume text.
"""

from typing import Optional, List, Tuple
import re

import spacy


class EducationExtractor:
    """
    Extract education information from resume text.
    
    Extracts:
    - Degree type (Bachelor's, Master's, PhD, etc.)
    - Field of study (Computer Science, Engineering, etc.)
    - Institution (University name)
    
    Usage:
        extractor = EducationExtractor()
        education = extractor.extract_education(text)
    """
    
    def __init__(self, use_nlp: bool = True):
        """
        Initialize the education extractor.
        
        Args:
            use_nlp: If True, use spaCy for institution detection.
        """
        self.use_nlp = use_nlp
        self._nlp = None
        
        if self.use_nlp:
            self._load_spacy_model()
        
        # Build degree patterns
        self._build_degree_patterns()
        
        # Build known institutions list
        self._build_known_institutions()
    
    def _load_spacy_model(self):
        """Load spaCy model lazily."""
        if self._nlp is not None:
            return
        
        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            ) from e

    def _build_degree_patterns(self):
        """
        Build regex patterns for degree detection.
        
        Patterns capture:
        - Degree type
        - Optional field of study
        """
        # Degree abbreviations and full names
        bachelor_variants = [
            r"bachelor'?s?\s+of",
            r"bachelor'?s?",
            r"b\.?\s?s\.?c?\.?",
            r"b\.?\s?a\.?",
            r"b\.?\s?tech\.?",
            r"b\.?\s?e\.?"
        ]
        
        master_variants = [
            r"master'?s?\s+of",
            r"master'?s?",
            r"m\.?\s?s\.?c?\.?",
            r"m\.?\s?a\.?",
            r"m\.?\s?b\.?\s?a\.?",
            r"m\.?\s?eng\.?",
            r"m\.?\s?tech\.?",
            r"m\.?\s?e\.?"
        ]
        
        phd_variants = [
            r"ph\.?\s?d\.?",
            r"doctorate",
            r"doctoral",
            r"d\.?\s?phil\.?"
        ]
        
        # Create patterns
        self.degree_patterns = []
        
        for degree_type, variants in [
            ('phd', phd_variants),      # Check PhD first
            ('master', master_variants), # Then Master
            ('bachelor', bachelor_variants) # Then Bachelor
        ]:
            # Pattern 1: "Bachelor of Science in Computer Science"
            pattern1 = (
                r'\b(' + '|'.join(variants) + r')\s+'
                r'(?:of\s+)?(?:science|arts|engineering|business|technology)?\s+'
                r'(?:in\s+)?'
                r'([A-Z][a-zA-Z\s&-]+?)'
                r'(?:\s+(?:from|at|,|\(|\n|$))'
            )
            self.degree_patterns.append((degree_type, pattern1))
            
            # Pattern 2: "MS in Computer Science" or "PhD in AI"
            pattern2 = (
                r'\b(' + '|'.join(variants) + r')\s+'
                r'in\s+'
                r'([A-Z][a-zA-Z\s&-]+?)'
                r'(?:\s+(?:from|at|,|\(|\n|$))'
            )
            self.degree_patterns.append((degree_type, pattern2))
            
            # Pattern 3: "MS Computer Science" (no "in")
            pattern3 = (
                r'\b(' + '|'.join(variants) + r')\s+'
                r'([A-Z][a-zA-Z\s&]+)'
                r'(?:,|\s+(?:from|at|\(|\n|$))'
            )
            self.degree_patterns.append((degree_type, pattern3))
        
        # Degree normalization
        self.degree_normalization = {
            'bachelor': 'Bachelor',
            'master': 'Master',
            'phd': 'PhD'
        }

    def _build_known_institutions(self):
        """
        Build list of known universities and keywords.
        
        Used to identify institutions even without full name.
        """
        # Common university keywords
        self.institution_keywords = [
            "university", "college", "institute", "school",
            "polytechnic", "academy"
        ]
        
        # Well-known institutions (can be expanded)
        self.known_institutions = {
            "mit", "stanford", "harvard", "berkeley", "caltech",
            "cmu", "carnegie mellon", "georgia tech", "ut austin",
            "oxford", "cambridge", "imperial college",
            "iit", "nit", "bits", "iiit", "nau", "northern arizona"
        }

    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand common abbreviations before pattern matching.
        
        This helps patterns catch abbreviated formats.
        """
        # Common degree abbreviations with context
        replacements = [
            (r'\bBS\s+CS\b', 'Bachelor of Science in Computer Science'),
            (r'\bMS\s+CS\b', 'Master of Science in Computer Science'),
            (r'\bBS\s+EE\b', 'Bachelor of Science in Electrical Engineering'),
            (r'\bMS\s+EE\b', 'Master of Science in Electrical Engineering'),
            (r'\bPh\.D\.\s+in\s+', 'PhD in '),
            (r'\bM\.S\.\s+in\s+', 'MS in '),
            (r'\bB\.S\.\s+in\s+', 'BS in '),
        ]
        
        expanded = text
        for pattern, replacement in replacements:
            expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
        
        return expanded

    def extract_education(self, text: str) -> str:
        """
        Extract education information from resume text.
        
        Returns formatted string: "[Degree] in [Field] | [Institution]"
        
        Args:
            text: Resume text
            
        Returns:
            Formatted education string, or empty string if not found
        """
        if not text or not text.strip():
            return ""
        
        # Expand abbreviations first
        expanded_text = self._expand_abbreviations(text)
        
        # Step 1: Extract degree and field
        degree_info = self._extract_degree_and_field(expanded_text)
        
        # Step 2: Extract institution (use original text)
        institution = self._extract_institution(text)
        
        # Step 3: Format result
        if degree_info or institution:
            return self._format_education(degree_info, institution)
        
        return ""

    def _extract_degree_and_field(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Extract degree type and field of study.
        
        Args:
            text: Resume text
            
        Returns:
            Tuple of (degree_type, field) or None
        """
        best_match = None
        best_priority = -1
        
        priority_map = {'phd': 3, 'master': 2, 'bachelor': 1}
        
        for degree_type, pattern in self.degree_patterns:
            try:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    if len(match.groups()) < 2:
                        continue
                    
                    degree_raw = match.group(1).strip()
                    field_raw = match.group(2).strip() if match.group(2) else ""
                    
                    # Clean field name
                    field = self._clean_field_name(field_raw)
                    
                    # Skip if field is too short or generic
                    if len(field) < 2:
                        continue
                    
                    # Skip generic/noise words
                    noise_words = {
                        'of', 'in', 'from', 'at', 'the', 'and', 'or',
                        'university', 'college', 'institute', 'school'
                    }
                    if field.lower() in noise_words:
                        continue
                    
                    # Normalize degree
                    degree = self.degree_normalization.get(degree_type, degree_type.title())
                    
                    # Check if this is a better match (higher priority)
                    current_priority = priority_map.get(degree_type, 0)
                    if current_priority > best_priority:
                        best_match = (degree, field)
                        best_priority = current_priority
            
            except Exception as e:
                # Continue if pattern fails
                continue
        
        return best_match
    
    def _clean_field_name(self, field: str) -> str:
        """
        Clean and standardize field of study name.
        
        Args:
            field: Raw field name
            
        Returns:
            Cleaned field name
        """
        if not field:
            return ""
        
        # Remove extra whitespace
        field = re.sub(r'\s+', ' ', field).strip()
        
        # Remove trailing punctuation
        field = re.sub(r'[,;.()]+$', '', field).strip()
        
        # Remove trailing noise words
        field = re.sub(r'\s+(from|at|the|and|or)$', '', field, flags=re.IGNORECASE).strip()
        
        # Remove leading noise
        field = re.sub(r'^(of|in|the)\s+', '', field, flags=re.IGNORECASE).strip()
        
        # Limit length (max 50 chars)
        if len(field) > 50:
            field = field[:50].rsplit(' ', 1)[0]
        
        # Title case
        field = field.title()
        
        # Common abbreviation expansion
        abbrev_map = {
            'Cs': 'Computer Science',
            'Ce': 'Computer Engineering',
            'Ee': 'Electrical Engineering',
            'Ece': 'Electrical And Computer Engineering',
            'Me': 'Mechanical Engineering',
            'Che': 'Chemical Engineering',
            'It': 'Information Technology',
            'Is': 'Information Systems',
            'Ai': 'Artificial Intelligence',
            'Ml': 'Machine Learning',
            'Ds': 'Data Science',
            'Mis': 'Management Information Systems'
        }
        
        # Check exact match
        if field in abbrev_map:
            return abbrev_map[field]
        
        # Check if field ends with abbreviation
        words = field.split()
        if words and words[-1] in abbrev_map:
            words[-1] = abbrev_map[words[-1]]
            field = ' '.join(words)
        
        return field

    def _extract_institution(self, text: str) -> Optional[str]:
        """
        Extract institution name from text.
        
        Uses spaCy NER and keyword matching.
        
        Args:
            text: Resume text
            
        Returns:
            Institution name or None
        """
        if not self.use_nlp:
            return self._extract_institution_simple(text)
        
        # Use spaCy to find organizations
        doc = self._nlp(text)
        
        candidates = []
        
        # Strategy 1: ORG entities with university keywords
        for ent in doc.ents:
            if ent.label_ == "ORG":
                org_text = ent.text.strip()
                org_lower = org_text.lower()
                
                # Check if contains university keywords
                has_keyword = any(keyword in org_lower for keyword in self.institution_keywords)
                
                # Check against known institutions
                is_known = any(known in org_lower for known in self.known_institutions)
                
                if has_keyword or is_known:
                    candidates.append(org_text)
        
        # Strategy 2: Explicit pattern matching
        # "from [institution]", "at [institution]"
        pattern = r'(?:from|at)\s+([A-Z][a-zA-Z\s]+(?:University|College|Institute|School|Academy))'
        matches = re.findall(pattern, text)
        candidates.extend(matches)
        
        # Return first valid candidate
        if candidates:
            return candidates[0]
        
        return None
    
    def _extract_institution_simple(self, text: str) -> Optional[str]:
        """
        Simple institution extraction without NLP.
        
        Uses regex patterns only.
        """
        # Pattern: University/College name
        pattern = r'([A-Z][a-zA-Z\s]+(?:University|College|Institute|School|Academy))'
        matches = re.findall(pattern, text)
        
        if matches:
            return matches[0].strip()
        
        return None

    def _format_education(
        self,
        degree_info: Optional[Tuple[str, str]],
        institution: Optional[str]
    ) -> str:
        """
        Format education information into standard string.
        
        Format: "[Degree] in [Field] | [Institution]"
        
        Args:
            degree_info: Tuple of (degree, field) or None
            institution: Institution name or None
            
        Returns:
            Formatted string
        """
        parts = []
        
        if degree_info:
            degree, field = degree_info
            if field:
                parts.append(f"{degree} in {field}")
            else:
                parts.append(degree)
        
        if institution:
            parts.append(institution)
        
        return " | ".join(parts) if parts else ""

    def extract_all_degrees(self, text: str) -> List[str]:
        """
        Extract all degrees mentioned in text (not just highest).
        
        Useful for debugging and analysis.
        
        Args:
            text: Resume text
            
        Returns:
            List of all degree strings found
        """
        expanded_text = self._expand_abbreviations(text)
        degrees = []
        
        for degree_type, pattern in self.degree_patterns:
            try:
                matches = re.finditer(pattern, expanded_text, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    if len(match.groups()) < 2:
                        continue
                    
                    degree_raw = match.group(1).strip()
                    field_raw = match.group(2).strip() if match.group(2) else ""
                    
                    field = self._clean_field_name(field_raw)
                    
                    # Skip noise
                    if len(field) < 2:
                        continue
                    
                    noise_words = {'of', 'in', 'from', 'at', 'the', 'university', 'college'}
                    if field.lower() in noise_words:
                        continue
                    
                    degree = self.degree_normalization.get(degree_type, degree_type.title())
                    
                    degree_str = f"{degree} in {field}" if field else degree
                    
                    # Avoid duplicates
                    if degree_str not in degrees:
                        degrees.append(degree_str)
            
            except Exception:
                continue
        
        return degrees
