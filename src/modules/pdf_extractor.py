# src/modules/pdf_extractor.py
"""
PDF Text Extraction Module
Extracts clean, structured text from PDF resume files.
"""

import re
from pathlib import Path
from typing import Optional
from PyPDF2 import PdfReader


class PDFExtractor:
    """
    Extract and clean text from PDF resume files.
    
    Features:
    - Layout-aware extraction (preserves column structure)
    - Text cleaning and normalization
    - Handles multi-page PDFs
    - Removes headers/footers and page numbers
    
    Usage:
        extractor = PDFExtractor()
        text = extractor.extract_text("resume.pdf")
    """
    
    def __init__(self):
        """Initialize the PDF extractor."""
        # Configuration for text cleaning
        self.max_consecutive_newlines = 2
        self.min_line_length = 2  # Ignore very short lines (likely noise)
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Cleaned text content
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            ValueError: If PDF is empty or corrupted
            
        Example:
            >>> extractor = PDFExtractor()
            >>> text = extractor.extract_text("resume.pdf")
            >>> print(text[:100])
            "JOHN DOE
            Senior Software Engineer
            ..."
        """
        # Validate file exists
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Extract raw text
        raw_text = self._extract_raw_text(pdf_file)
        
        # Clean and normalize
        cleaned_text = self._clean_text(raw_text)
        
        return cleaned_text


    def _extract_raw_text(self, pdf_file: Path) -> str:
        """
        Extract raw text from PDF using PyPDF2.
        
        Uses layout mode to preserve spatial structure (PyPDF2 3.0+).
        Falls back to simple mode for older versions.
        Combines text from all pages with page breaks.
        
        Args:
            pdf_file: Path object to PDF file
            
        Returns:
            Raw extracted text
            
        Raises:
            ValueError: If PDF is corrupted or empty
        """
        try:
            # Open PDF file
            reader = PdfReader(str(pdf_file))
            
            # Check if PDF has pages
            num_pages = len(reader.pages)
            if num_pages == 0:
                raise ValueError("PDF file is empty (0 pages)")
            
            # Extract text from each page
            full_text = []
            for page_num, page in enumerate(reader.pages, start=1):
                # Extract with layout mode (if supported)
                try:
                    # PyPDF2 3.0+ with layout preservation
                    page_text = page.extract_text(extraction_mode="layout")
                except TypeError:
                    # Older PyPDF2 - fallback to simple extraction
                    page_text = page.extract_text()
                
                if page_text and page_text.strip():
                    full_text.append(page_text)
                    
                    # Add page separator (except for last page)
                    if page_num < num_pages:
                        full_text.append("\n\n--- PAGE BREAK ---\n\n")
            
            # Combine all pages
            combined_text = "".join(full_text)
            
            if not combined_text.strip():
                raise ValueError("PDF appears to be empty (no text extracted)")
            
            return combined_text
            
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error reading PDF: {str(e)}")



    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.
        
        Cleaning steps:
        1. Remove page breaks markers
        2. Normalize whitespace
        3. Remove page numbers
        4. Remove excessive newlines
        5. Remove very short lines (noise)
        6. Normalize bullet points
        7. Remove leading/trailing whitespace
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text
        """
        # Step 1: Remove page break markers
        text = re.sub(r'\n+---\s*PAGE BREAK\s*---\n+', '\n\n', text)
        
        # Step 2: Normalize horizontal whitespace (spaces/tabs)
        # Replace multiple spaces/tabs with single space
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Step 3: Remove common page number patterns
        # Examples: "Page 1", "1 of 2", "- 2 -"
        text = re.sub(r'\n\s*(?:Page\s+)?\d+\s*(?:of\s+\d+)?\s*\n', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'\n\s*[-–—]\s*\d+\s*[-–—]\s*\n', '\n', text)
        
        # Step 4: Remove excessive newlines (max 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Step 5: Remove very short lines (likely noise)
        # Keep lines that are at least min_line_length characters
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep empty lines (for structure) or lines with enough content
            if len(stripped) == 0 or len(stripped) >= self.min_line_length:
                cleaned_lines.append(line)
        text = '\n'.join(cleaned_lines)
        
        # Step 6: Normalize bullet points and special characters
        # Convert various bullet styles to standard "-"
        text = re.sub(r'[•●○■□▪▫⦿⦾]', '-', text)
        
        # Step 7: Clean up spacing around bullets
        text = re.sub(r'\n\s*-\s+', '\n- ', text)
        
        # Step 8: Final cleanup - remove leading/trailing whitespace
        text = text.strip()
        
        # Step 9: Final excessive newline check
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text


    def get_text_statistics(self, text: str) -> dict:
        """
        Get statistics about extracted text.
        
        Useful for debugging and validation.
        
        Args:
            text: Extracted text
            
        Returns:
            Dictionary with statistics
        """
        lines = text.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        words = text.split()
        
        return {
            "total_characters": len(text),
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "total_words": len(words),
            "average_line_length": sum(len(line) for line in non_empty_lines) / len(non_empty_lines) if non_empty_lines else 0,
            "has_content": len(text.strip()) > 0
        }


    def validate_extraction(self, text: str) -> tuple[bool, str]:
        """
        Validate that extraction was successful.
        
        Checks:
        - Text is not empty
        - Text has minimum length (50 chars)
        - Text has reasonable word count (10+ words)
        
        Args:
            text: Extracted text
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Example:
            >>> is_valid, error = extractor.validate_extraction(text)
            >>> if not is_valid:
            >>>     print(f"Extraction failed: {error}")
        """
        # Check 1: Not empty
        if not text or not text.strip():
            return False, "Extracted text is empty"
        
        # Check 2: Minimum length
        if len(text.strip()) < 50:
            return False, f"Text too short ({len(text)} chars, minimum 50)"
        
        # Check 3: Reasonable word count
        words = text.split()
        if len(words) < 10:
            return False, f"Too few words ({len(words)}, minimum 10)"
        
        # All checks passed
        return True, "Extraction successful"


    def extract_from_directory(self, directory_path: str, pattern: str = "*.pdf") -> dict[str, str]:
        """
        Extract text from all PDFs in a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            pattern: File pattern to match (default: "*.pdf")
            
        Returns:
            Dictionary mapping filename to extracted text
            
        Example:
            >>> extractor = PDFExtractor()
            >>> results = extractor.extract_from_directory("data/resumes/")
            >>> for filename, text in results.items():
            >>>     print(f"{filename}: {len(text)} characters")
        """
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory.is_dir():
            raise ValueError(f"Not a directory: {directory_path}")
        
        results = {}
        pdf_files = list(directory.glob(pattern))
        
        if not pdf_files:
            print(f"Warning: No PDF files found matching pattern '{pattern}'")
            return results
        
        for pdf_file in pdf_files:
            try:
                text = self.extract_text(str(pdf_file))
                results[pdf_file.name] = text
                print(f"✓ Extracted: {pdf_file.name} ({len(text)} chars)")
            except Exception as e:
                print(f"✗ Failed: {pdf_file.name} - {str(e)}")
                results[pdf_file.name] = None
        
        return results
