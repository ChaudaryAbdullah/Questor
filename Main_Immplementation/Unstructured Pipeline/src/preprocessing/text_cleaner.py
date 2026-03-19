"""
Text Cleaner Module for SEC 10-K Documents
Provides utilities for cleaning and normalizing financial document text.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..utils.logger import get_preprocessing_logger


@dataclass
class CleaningResult:
    """Result of text cleaning operation."""
    cleaned_text: str
    changes_made: List[str]
    original_length: int
    cleaned_length: int


class TextCleaner:
    """
    Text cleaning utilities for SEC filings.
    Handles normalization, whitespace cleanup, and noise removal.
    """
    
    def __init__(self):
        self.logger = get_preprocessing_logger()
        
        # Patterns for header/footer detection
        self.header_patterns = [
            r"^Page\s+\d+\s*(of\s*\d+)?.*$",
            r"^-\s*\d+\s*-.*$",
            r"^\d+$",  # Standalone page numbers
            r"^Table of Contents.*$",
            r"^INDEX.*$",
            r"^PART\s+[IVX]+.*$",  # At start of line, likely header
        ]
        
        self.footer_patterns = [
            r"^See accompanying notes.*$",
            r"^The accompanying notes are.*$",
            r"^F-\d+$",  # Financial statement page markers
            r"^S-\d+$",  # Schedule page markers
        ]
        
        # Compile regex patterns
        self.header_regex = [re.compile(p, re.IGNORECASE | re.MULTILINE) 
                           for p in self.header_patterns]
        self.footer_regex = [re.compile(p, re.IGNORECASE | re.MULTILINE) 
                           for p in self.footer_patterns]
        
        # SEC-specific noise patterns
        self.noise_patterns = {
            "xml_tags": re.compile(r"<[^>]+>"),
            "html_entities": re.compile(r"&[a-zA-Z]+;|&#\d+;"),
            "repeated_chars": re.compile(r"(.)\1{4,}"),
            "form_markers": re.compile(r"^\s*FORM\s+10-K\s*$", re.MULTILINE),
            "table_artifacts": re.compile(r"[\|\+\-]{3,}"),
            "excessive_dots": re.compile(r"\.{4,}"),
            "filing_metadata": re.compile(
                r"(Commission file number|CIK|ACCESSION NUMBER):?\s*[\d\-]+",
                re.IGNORECASE
            ),
        }
    
    def clean(self, text: str, options: Optional[Dict] = None) -> CleaningResult:
        """
        Main cleaning method that applies all cleaning steps.
        
        Args:
            text: Raw text to clean
            options: Optional dictionary with cleaning options
        
        Returns:
            CleaningResult with cleaned text and metadata
        """
        if options is None:
            options = {
                "normalize_unicode": True,
                "remove_headers_footers": True,
                "remove_noise": True,
                "normalize_whitespace": True,
                "remove_empty_lines": True
            }
        
        original_length = len(text)
        changes_made = []
        cleaned = text
        
        # Step 1: Unicode normalization
        if options.get("normalize_unicode", True):
            cleaned = self.normalize_unicode(cleaned)
            changes_made.append("Normalized Unicode")
        
        # Step 2: Remove headers and footers
        if options.get("remove_headers_footers", True):
            cleaned = self.remove_headers_footers(cleaned)
            changes_made.append("Removed headers/footers")
        
        # Step 3: Remove noise patterns
        if options.get("remove_noise", True):
            cleaned = self.remove_noise(cleaned)
            changes_made.append("Removed noise patterns")
        
        # Step 4: Normalize whitespace
        if options.get("normalize_whitespace", True):
            cleaned = self.normalize_whitespace(cleaned)
            changes_made.append("Normalized whitespace")
        
        # Step 5: Remove excessive empty lines
        if options.get("remove_empty_lines", True):
            cleaned = self.remove_excessive_empty_lines(cleaned)
            changes_made.append("Removed excessive empty lines")
        
        return CleaningResult(
            cleaned_text=cleaned,
            changes_made=changes_made,
            original_length=original_length,
            cleaned_length=len(cleaned)
        )
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters to standard form.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        # Normalize to NFC form
        text = unicodedata.normalize("NFC", text)
        
        # Replace common problematic characters
        replacements = {
            "\u2019": "'",  # Right single quote
            "\u2018": "'",  # Left single quote
            "\u201c": '"',  # Left double quote
            "\u201d": '"',  # Right double quote
            "\u2014": "-",  # Em dash
            "\u2013": "-",  # En dash
            "\u00a0": " ",  # Non-breaking space
            "\u2026": "...",  # Ellipsis
            "\u00b7": "*",  # Middle dot
            "\u2022": "*",  # Bullet
            "\u00ae": "(R)",  # Registered trademark
            "\u2122": "(TM)",  # Trademark
            "\u00a9": "(C)",  # Copyright
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def remove_headers_footers(self, text: str) -> str:
        """
        Remove common header and footer patterns from SEC filings.
        
        Args:
            text: Input text
        
        Returns:
            Text with headers/footers removed
        """
        lines = text.split("\n")
        cleaned_lines = []
        
        for line in lines:
            is_header_footer = False
            
            # Check header patterns
            for pattern in self.header_regex:
                if pattern.match(line.strip()):
                    is_header_footer = True
                    break
            
            # Check footer patterns
            if not is_header_footer:
                for pattern in self.footer_regex:
                    if pattern.match(line.strip()):
                        is_header_footer = True
                        break
            
            if not is_header_footer:
                cleaned_lines.append(line)
        
        return "\n".join(cleaned_lines)
    
    def remove_noise(self, text: str) -> str:
        """
        Remove SEC-specific noise patterns.
        
        Args:
            text: Input text
        
        Returns:
            Text with noise removed
        """
        cleaned = text
        
        for name, pattern in self.noise_patterns.items():
            if name in ["repeated_chars"]:
                # For repeated chars, replace with single instance
                cleaned = pattern.sub(r"\1", cleaned)
            else:
                cleaned = pattern.sub(" ", cleaned)
        
        return cleaned
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Normalize whitespace characters.
        
        Args:
            text: Input text
        
        Returns:
            Text with normalized whitespace
        """
        # Replace tabs with spaces
        text = text.replace("\t", "    ")
        
        # Normalize multiple spaces to single space
        text = re.sub(r" {2,}", " ", text)
        
        # Remove trailing whitespace from each line
        lines = [line.rstrip() for line in text.split("\n")]
        
        return "\n".join(lines)
    
    def remove_excessive_empty_lines(self, text: str) -> str:
        """
        Reduce multiple consecutive empty lines to maximum two.
        
        Args:
            text: Input text
        
        Returns:
            Text with reduced empty lines
        """
        # Replace 3+ consecutive newlines with 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text: Input text
        
        Returns:
            List of sentences
        """
        # Simple sentence tokenization
        # Handles abbreviations and decimal numbers
        sentence_pattern = r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+"
        sentences = re.split(sentence_pattern, text)
        
        # Clean up sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def detect_section_break(self, text: str) -> bool:
        """
        Detect if text represents a section break.
        
        Args:
            text: Input text line
        
        Returns:
            True if text is a section break
        """
        section_patterns = [
            r"^ITEM\s+\d+[A-Z]?\.",
            r"^PART\s+[IVX]+\s*[-–—]",
            r"^NOTE\s+\d+\s*[-–—:]",
            r"^={3,}$",
            r"^-{3,}$",
        ]
        
        text = text.strip()
        for pattern in section_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def remove_boilerplate(self, text: str) -> str:
        """
        Remove common SEC filing boilerplate text.
        
        Args:
            text: Input text
        
        Returns:
            Text with boilerplate removed
        """
        boilerplate_patterns = [
            r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION.*?Washington,?\s*D\.?C\.?\s*\d+",
            r"For the fiscal year ended.*?Commission [Ff]ile [Nn]umber.*?\d+-\d+",
            r"Securities registered pursuant to Section.*?Securities Act of 1933",
            r"Indicate by check mark whether the registrant.*?Yes\s*[\[\]xX\s]*No",
        ]
        
        cleaned = text
        for pattern in boilerplate_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        
        return cleaned
    
    def get_text_statistics(self, text: str) -> Dict:
        """
        Get statistics about the text.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with text statistics
        """
        lines = text.split("\n")
        words = text.split()
        sentences = self.extract_sentences(text)
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "sentence_count": len(sentences),
            "average_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "average_sentence_length": len(words) / len(sentences) if sentences else 0,
            "empty_line_count": sum(1 for line in lines if not line.strip()),
        }
