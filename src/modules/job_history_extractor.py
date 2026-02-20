# src/modules/job_history_extractor.py
"""
Job History Extraction Module
Extracts structured job history (title, company, dates, duration) from resume text.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import re
from datetime import datetime

from dateutil.relativedelta import relativedelta
import spacy


@dataclass
class JobHistoryEntry:
    title: str
    company: Optional[str]
    start_date: Optional[datetime]
    end_date: Optional[datetime]
    duration_years: Optional[float]

    def format(self) -> str:
        """
        Format as a human-readable string:
        "Title at Company (Mon YYYY – Mon YYYY, X.Y years)"
        """
        parts = []

        # Title + company
        if self.company:
            parts.append(f"{self.title} at {self.company}")
        else:
            parts.append(self.title)

        # Dates
        date_part = ""
        if self.start_date and self.end_date:
            start_str = self.start_date.strftime("%b %Y")
            end_str = self.end_date.strftime("%b %Y")
            date_part = f"{start_str} – {end_str}"
        elif self.start_date:
            start_str = self.start_date.strftime("%b %Y")
            date_part = f"{start_str} – Present"

        # Duration
        if self.duration_years is not None and self.duration_years > 0:
            duration_str = f"{self.duration_years:.1f} years"
            if date_part:
                parts.append(f"({date_part}, {duration_str})")
            else:
                parts.append(f"({duration_str})")
        elif date_part:
            parts.append(f"({date_part})")

        return " ".join(parts)


class JobHistoryExtractor:
    """
    Extract job history entries from resume text.

    For each job, tries to extract:
    - Title
    - Company
    - Start/end dates
    - Duration in years
    """

    def __init__(self, use_nlp: bool = True, current_date: Optional[datetime] = None):
        self.use_nlp = use_nlp
        self.current_date = current_date or datetime.now()
        self._nlp = None

        if self.use_nlp:
            self._load_spacy_model()

        # Month mapping
        self.month_map = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }

    def _load_spacy_model(self):
        if self._nlp is not None:
            return

        try:
            self._nlp = spacy.load("en_core_web_sm")
        except OSError as e:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            ) from e

    def extract_job_history(self, text: str) -> List[JobHistoryEntry]:
        """
        Extract ordered job history from resume text.

        Args:
            text: Full resume text

        Returns:
            List of JobHistoryEntry (most recent first if possible)
        """
        if not text or not text.strip():
            return []

        lines = [line.strip() for line in text.splitlines()]
        # Remove empty lines but keep index mapping
        indexed_lines = [(i, line) for i, line in enumerate(lines) if line]

        # 1. Find date range lines
        date_lines = self._find_date_lines(indexed_lines)

        # 2. For each date line, find context (title + company)
        entries: List[JobHistoryEntry] = []
        for idx, (line_index, date_match) in enumerate(date_lines):
            start_date, end_date = self._parse_date_match(date_match)
            if not start_date or not end_date:
                continue

            # Look back 1–2 non-empty lines for title/company
            title, company = self._infer_title_and_company(
                lines, line_index, window=2
            )

            if not title:
                continue

            duration_years = self._compute_duration_years(start_date, end_date)

            entry = JobHistoryEntry(
                title=title,
                company=company,
                start_date=start_date,
                end_date=end_date,
                duration_years=duration_years,
            )
            entries.append(entry)

        # 3. Sort by start_date descending (most recent first)
        entries.sort(
            key=lambda e: e.start_date or datetime(1900, 1, 1),
            reverse=True
        )
        return entries

    def _find_date_lines(self, indexed_lines: List[Tuple[int, str]]) -> List[Tuple[int, re.Match]]:
        """
        Find lines that contain date ranges.

        Returns:
            List of (line_index, regex_match)
        """
        date_lines = []

        # Pattern: "Jan 2020 - Mar 2023" or "January 2020 – Present"
        pattern_month = re.compile(
            r'(\b\w+\b)\s+(\d{4})\s*[-–—]\s*(\b\w+\b|\bpresent\b|\bcurrent\b|\bnow\b)\s*(\d{4})?',
            re.IGNORECASE,
        )

        # Pattern: "2018 - 2023" or "2018 - Present"
        pattern_year = re.compile(
            r'(\d{4})\s*[-–—]\s*(\d{4}|\bpresent\b|\bcurrent\b|\bnow\b)',
            re.IGNORECASE,
        )

        for idx, line in indexed_lines:
            m1 = pattern_month.search(line)
            if m1:
                date_lines.append((idx, m1))
                continue

            m2 = pattern_year.search(line)
            if m2:
                date_lines.append((idx, m2))

        return date_lines

    def _parse_date_match(self, match: re.Match) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Parse regex match into (start_date, end_date).
        Handles both month-year and year-only patterns.
        """
        groups = match.groups()

        # Month-year pattern: (start_month, start_year, end_month_or_word, end_year_optional)
        if len(groups) == 4:
            start_month, start_year, end_part, end_year_optional = groups
            start_date = self._parse_month_year(start_month, start_year)

            # End: could be month+year or a word like "present"
            end_str = end_part.lower()
            if end_str in {"present", "current", "now"}:
                end_date = self.current_date
            else:
                # If a separate end year exists, use it; else treat end_part as month and reuse start_year
                end_year = end_year_optional or start_year
                end_date = self._parse_month_year(end_part, end_year)

            return start_date, end_date

        # Year-only pattern: (start_year, end_year_or_word)
        if len(groups) == 2:
            start_year, end_part = groups
            try:
                start_y = int(start_year)
            except ValueError:
                return None, None

            if start_y < 1970 or start_y > self.current_date.year:
                return None, None

            start_date = datetime(start_y, 1, 1)

            end_str = end_part.lower()
            if end_str in {"present", "current", "now"}:
                end_date = self.current_date
            else:
                try:
                    end_y = int(end_str)
                except ValueError:
                    return None, None
                # Sanity: not too far in future
                if end_y < start_y or end_y > self.current_date.year + 1:
                    return None, None
                end_date = datetime(end_y, 12, 31)

            return start_date, end_date

        return None, None

    def _parse_month_year(self, month_str: str, year_str: str) -> Optional[datetime]:
        """Parse strings like ('Jan', '2020') into datetime(2020, 1, 1)."""
        try:
            year = int(year_str)
        except ValueError:
            return None

        month_key = month_str.lower().strip()
        month = self.month_map.get(month_key)
        if not month:
            return None

        if year < 1970 or year > self.current_date.year + 1:
            return None

        return datetime(year, month, 1)

    def _infer_title_and_company(self, lines: List[str], date_line_index: int, window: int = 2) -> Tuple[Optional[str], Optional[str]]:
        """
        Look at lines above the date line to infer title and company.

        Args:
            lines: All lines of text
            date_line_index: Index of the line containing date range
            window: How many lines above to consider

        Returns:
            (title, company)
        """
        start_idx = max(0, date_line_index - window)
        candidate_lines = [lines[i].strip() for i in range(start_idx, date_line_index) if lines[i].strip()]

        if not candidate_lines:
            return None, None

        # Heuristic: last non-empty line before dates is usually the title/company line
        context = " ".join(candidate_lines[-2:])  # up to 2 lines combined
        title = candidate_lines[-1]

        company = None
        if self.use_nlp and self._nlp is not None:
            company = self._extract_company_with_nlp(context)

        # If NLP fails, try simple heuristic:
        if not company:
            company = self._extract_company_simple(context)

        # Clean title (remove trailing punctuation, location, etc.)
        title = self._clean_title(title, company)

        return title if title else None, company

    def _extract_company_with_nlp(self, text: str) -> Optional[str]:
        """Use spaCy ORG entities to infer company."""
        doc = self._nlp(text)
        orgs = [ent.text.strip() for ent in doc.ents if ent.label_ == "ORG"]
        return orgs[0] if orgs else None

    def _extract_company_simple(self, text: str) -> Optional[str]:
        """
        Simple heuristic: look for patterns like 'at X', 'X, Inc', 'X Ltd'.
        """
        # Pattern: 'at Company Name'
        m = re.search(r'\bat\s+([A-Z][A-Za-z0-9&.\s]+)', text)
        if m:
            return m.group(1).strip()

        # Pattern: 'Company Name, Inc.' or 'Company Name Ltd'
        m = re.search(r'([A-Z][A-Za-z0-9&.\s]+?\s+(?:Inc\.?|LLC|Ltd\.?|GmbH|Corp\.?))', text)
        if m:
            return m.group(1).strip()

        # Fallback: if there's a comma, assume after comma is company
        if "," in text:
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if len(parts) >= 2:
                # e.g., "Senior Engineer, Google" → company = "Google"
                return parts[-1]

        return None

    def _clean_title(self, title: str, company: Optional[str]) -> str:
        """Remove company name and extra punctuation from title string."""
        t = title.strip()

        if company and company in t:
            t = t.replace(company, "").strip(",|- ")

        # Remove trailing location if separated by '|'
        if "|" in t:
            t = t.split("|")[0].strip()

        # Basic cleanup
        t = re.sub(r'\s+', ' ', t)
        return t

    def _compute_duration_years(self, start_date: datetime, end_date: datetime) -> float:
        """Compute duration in years with month precision."""
        if end_date < start_date:
            return 0.0

        delta = relativedelta(end_date, start_date)
        years = delta.years + delta.months / 12.0
        return round(years, 1)
