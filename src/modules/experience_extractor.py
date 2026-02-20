# src/modules/experience_extractor.py
"""
Experience Extraction Module
Extracts years of work experience from resume text.
"""

from typing import List, Tuple, Optional
import re
from datetime import datetime
from dateutil import parser as date_parser
from dateutil.relativedelta import relativedelta


class ExperienceExtractor:
    """
    Extract years of work experience from resume text.
    
    Strategies:
    1. Direct mentions ("5 years of experience")
    2. Date range calculations ("2018-2023" → 5 years)
    3. Aggregation of multiple positions
    
    Usage:
        extractor = ExperienceExtractor()
        years = extractor.extract_years_of_experience(text)
    """
    
    def __init__(self, current_date: Optional[datetime] = None):
        """
        Initialize the experience extractor.
        
        Args:
            current_date: Reference date for "Present" calculations.
                         Defaults to today.
        """
        self.current_date = current_date or datetime.now()
        
        # Month name mappings
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

    def extract_years_of_experience(self, text: str) -> float:
        """
        Extract total years of work experience from resume text.
        
        Strategy:
        1. Look for direct mentions ("5 years of experience")
        2. Extract and calculate date ranges
        3. Return the maximum value found
        
        Args:
            text: Resume text
            
        Returns:
            Years of experience as float (e.g., 5.5)
            Returns 0.0 if no experience found
            
        Example:
            >>> extractor = ExperienceExtractor()
            >>> years = extractor.extract_years_of_experience(resume_text)
            >>> print(f"{years:.1f} years")
            "5.5 years"
        """
        if not text or not text.strip():
            return 0.0
        
        # Strategy 1: Direct mentions
        direct_years = self._extract_direct_mention(text)
        
        # Strategy 2: Date range calculations
        date_ranges = self._extract_date_ranges(text)
        calculated_years = self._calculate_total_experience(date_ranges)
        
        # Return maximum (most reliable estimate)
        return max(direct_years, calculated_years)

    def _extract_direct_mention(self, text: str) -> float:
        """
        Extract years from direct mentions like "5 years of experience".
        
        Patterns:
        - "X years of experience"
        - "X+ years"
        - "Over X years"
        - "X years working"
        
        Args:
            text: Resume text
            
        Returns:
            Years mentioned, or 0.0 if not found
        """
        patterns = [
            r'(\d+\.?\d*)\+?\s*years?\s+of\s+(?:work\s+)?experience',
            r'(\d+\.?\d*)\+?\s*years?\s+(?:working|in)',
            r'over\s+(\d+\.?\d*)\s*years?',
            r'more\s+than\s+(\d+\.?\d*)\s*years?',
            r'(\d+\.?\d*)\+\s*years?'
        ]
        
        found_years = []
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    years = float(match)
                    if 0 < years <= 50:  # Sanity check
                        found_years.append(years)
                except ValueError:
                    continue
        
        return max(found_years) if found_years else 0.0

    def _extract_date_ranges(self, text: str) -> List[Tuple[datetime, datetime]]:
        """
        Extract date ranges from text.
        
        Patterns:
        - "2018 - 2023"
        - "Jan 2020 - Mar 2023"
        - "January 2018 - Present"
        - "2020-Present"
        
        Args:
            text: Resume text
            
        Returns:
            List of (start_date, end_date) tuples
        """
        date_ranges = []
        
        # Pattern 1: "Month YYYY - Month YYYY"
        # Example: "Jan 2020 - Mar 2023"
        pattern1 = r'(\w+)\s+(\d{4})\s*[-–—]\s*(\w+)\s+(\d{4})'
        matches1 = re.findall(pattern1, text, re.IGNORECASE)
        
        for match in matches1:
            start_month, start_year, end_month, end_year = match
            start_date = self._parse_month_year(start_month, start_year)
            end_date = self._parse_month_year(end_month, end_year)
            
            if start_date and end_date:
                date_ranges.append((start_date, end_date))
        
        # Pattern 2: "Month YYYY - Present"
        pattern2 = r'(\w+)\s+(\d{4})\s*[-–—]\s*(present|current|now)'
        matches2 = re.findall(pattern2, text, re.IGNORECASE)
        
        for match in matches2:
            start_month, start_year, _ = match
            start_date = self._parse_month_year(start_month, start_year)
            
            if start_date:
                date_ranges.append((start_date, self.current_date))
        
        # Pattern 3: "YYYY - YYYY"
        # Example: "2018 - 2023"
        pattern3 = r'(\d{4})\s*[-–—]\s*(\d{4})'
        matches3 = re.findall(pattern3, text)
        
        for match in matches3:
            start_year, end_year = match
            try:
                start_date = datetime(int(start_year), 1, 1)
                end_date = datetime(int(end_year), 12, 31)
                
                if 1970 <= start_date.year <= self.current_date.year:
                    date_ranges.append((start_date, end_date))
            except ValueError:
                continue
        
        # Pattern 4: "YYYY - Present"
        pattern4 = r'(\d{4})\s*[-–—]\s*(present|current|now)'
        matches4 = re.findall(pattern4, text, re.IGNORECASE)
        
        for match in matches4:
            start_year, _ = match
            try:
                start_date = datetime(int(start_year), 1, 1)
                if 1970 <= start_date.year <= self.current_date.year:
                    date_ranges.append((start_date, self.current_date))
            except ValueError:
                continue
        
        return date_ranges

    def _parse_month_year(self, month_str: str, year_str: str) -> Optional[datetime]:
        """
        Parse month and year strings into datetime.
        
        Args:
            month_str: Month name or abbreviation
            year_str: Year as string
            
        Returns:
            datetime object or None if parsing fails
        """
        try:
            year = int(year_str)
            month_lower = month_str.lower().strip()
            
            # Try to get month from map
            month = self.month_map.get(month_lower)
            
            if month and 1970 <= year <= self.current_date.year + 1:
                return datetime(year, month, 1)
        except (ValueError, AttributeError):
            pass
        
        return None

    def _calculate_total_experience(self, date_ranges: List[Tuple[datetime, datetime]]) -> float:
        """
        Calculate total years from date ranges.
        
        To avoid counting overlapping periods twice, we take the
        maximum single range duration rather than summing.
        
        Args:
            date_ranges: List of (start_date, end_date) tuples
            
        Returns:
            Maximum years from any single range
        """
        if not date_ranges:
            return 0.0
        
        max_years = 0.0
        
        for start_date, end_date in date_ranges:
            if end_date < start_date:
                continue
            
            # Calculate difference with month precision
            delta = relativedelta(end_date, start_date)
            years = delta.years + (delta.months / 12.0)
            
            max_years = max(max_years, years)
        
        return round(max_years, 1)

    def extract_all_date_ranges(self, text: str) -> List[dict]:
        """
        Extract all date ranges with details (for debugging).
        
        Args:
            text: Resume text
            
        Returns:
            List of dictionaries with range details
        """
        date_ranges = self._extract_date_ranges(text)
        
        results = []
        for start_date, end_date in date_ranges:
            delta = relativedelta(end_date, start_date)
            years = delta.years + (delta.months / 12.0)
            
            results.append({
                'start': start_date.strftime('%B %Y'),
                'end': end_date.strftime('%B %Y'),
                'years': round(years, 1)
            })
        
        return results
