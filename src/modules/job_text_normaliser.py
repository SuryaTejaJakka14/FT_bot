# src/modules/job_text_normalizer.py
"""
JobTextNormalizer: Cleans and normalizes raw job description text
before passing it to the JobRequirementsExtractor.

Handles:
  - Whitespace normalization
  - Unicode/special character cleaning
  - HTML tag removal
  - Bullet point standardization
  - Line deduplication
"""

import re
import logging
from typing import List

logger = logging.getLogger(__name__)


class JobTextNormalizer:
    """
    Cleans and normalizes raw job description text.

    Designed to handle JD text from multiple sources:
      - Copy-pasted from job boards (LinkedIn, Indeed, Glassdoor)
      - Scraped HTML pages
      - PDF-extracted text
      - Manual input

    Usage:
        normalizer = JobTextNormalizer()
        clean_text = normalizer.normalize(raw_jd_text)
    """

    # Bullet point variants to standardize
    BULLET_PATTERNS = [
        r"^[\•\●\◆\▪\■\○\◦\‣\⁃\-\*]\s+",   # symbol bullets
        r"^\d+[\.\)]\s+",                      # numbered lists: 1. or 1)
    ]

    # HTML tag pattern
    HTML_TAG_PATTERN = re.compile(r"<[^>]+>")

    # Multiple whitespace/newlines
    MULTI_SPACE_PATTERN   = re.compile(r"[ \t]+")
    MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")

    # Unicode dashes/quotes to normalize
    UNICODE_REPLACEMENTS = [
        (r"[\u2013\u2014\u2015]", "-"),   # em/en dashes → hyphen
        (r"[\u2018\u2019]",       "'"),   # curly single quotes → straight
        (r"[\u201c\u201d]",       '"'),   # curly double quotes → straight
        (r"[\u00a0]",             " "),   # non-breaking space → regular space
        (r"[\u2022\u2023\u25e6]", "-"),   # bullet variants → hyphen
    ]

    def normalize(self, text: str) -> str:
        """
        Run the full normalization pipeline on raw JD text.

        Steps:
          1. Remove HTML tags
          2. Normalize unicode characters
          3. Standardize bullet points
          4. Normalize whitespace
          5. Remove duplicate lines
          6. Final strip

        Args:
            text: Raw job description text

        Returns:
            Cleaned, normalized text ready for extraction

        Example:
            >>> normalizer = JobTextNormalizer()
            >>> clean = normalizer.normalize("  Senior  Dev\\n\\n\\n• Python ")
            >>> print(clean)
            'Senior Dev\\n- Python'
        """
        if not text or not text.strip():
            return ""

        text = self._remove_html(text)
        text = self._normalize_unicode(text)
        text = self._normalize_bullets(text)
        text = self._normalize_whitespace(text)
        text = self._deduplicate_lines(text)

        return text.strip()

    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    def _remove_html(self, text: str) -> str:
        """Remove HTML tags and decode common HTML entities."""
        text = self.HTML_TAG_PATTERN.sub(" ", text)
        # Common HTML entities
        html_entities = [
            ("&amp;",  "&"),
            ("&lt;",   "<"),
            ("&gt;",   ">"),
            ("&nbsp;", " "),
            ("&quot;", '"'),
            ("&#39;",  "'"),
        ]
        for entity, replacement in html_entities:
            text = text.replace(entity, replacement)
        return text

    def _normalize_unicode(self, text: str) -> str:
        """Replace unicode variants with ASCII equivalents."""
        for pattern, replacement in self.UNICODE_REPLACEMENTS:
            text = re.sub(pattern, replacement, text)
        return text

    def _normalize_bullets(self, text: str) -> str:
        """Standardize all bullet point styles to '- '."""
        lines = text.split("\n")
        normalized = []
        for line in lines:
            stripped = line.strip()
            for pattern in self.BULLET_PATTERNS:
                if re.match(pattern, stripped):
                    stripped = re.sub(pattern, "- ", stripped)
                    break
            normalized.append(stripped)
        return "\n".join(normalized)

    def _normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces/tabs and excessive newlines."""
        text = self.MULTI_SPACE_PATTERN.sub(" ", text)
        text = self.MULTI_NEWLINE_PATTERN.sub("\n\n", text)
        return text

    def _deduplicate_lines(self, text: str) -> str:
        """
        Remove consecutive duplicate lines (e.g. repeated section headers).
        Preserves blank lines as section separators.
        """
        lines = text.split("\n")
        deduped: List[str] = []
        prev = None
        for line in lines:
            stripped = line.strip()
            # Always keep blank lines and non-duplicate lines
            if stripped == "" or stripped != prev:
                deduped.append(line)
            prev = stripped if stripped else prev
        return "\n".join(deduped)
