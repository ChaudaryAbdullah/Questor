"""
Document Processor Module for SEC 10-K Filings
Handles document ingestion, section extraction, and chunking.
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator
from datetime import datetime
import json

from .text_cleaner import TextCleaner, CleaningResult
from ..utils.logger import get_preprocessing_logger
from ..utils.config_manager import get_config


@dataclass
class TextChunk:
    """Represents a chunk of processed text with metadata."""
    id: str
    text: str
    section_type: str
    section_name: str
    page_number: Optional[int]
    chunk_index: int
    start_position: int
    end_position: int
    word_count: int
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "id": self.id,
            "text": self.text,
            "section_type": self.section_type,
            "section_name": self.section_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "word_count": self.word_count,
            "metadata": self.metadata
        }


@dataclass
class Section:
    """Represents an extracted section from the document."""
    name: str
    section_type: str
    content: str
    start_position: int
    end_position: int
    fraud_relevance: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "section_type": self.section_type,
            "content_length": len(self.content),
            "start_position": self.start_position,
            "end_position": self.end_position,
            "fraud_relevance": self.fraud_relevance
        }


@dataclass
class ProcessedDocument:
    """Represents a fully processed document."""
    filename: str
    file_path: str
    company_name: Optional[str]
    fiscal_year: Optional[str]
    filing_date: Optional[str]
    sections: List[Section]
    chunks: List[TextChunk]
    raw_text: str
    cleaned_text: str
    processing_time: float
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            "filename": self.filename,
            "file_path": self.file_path,
            "company_name": self.company_name,
            "fiscal_year": self.fiscal_year,
            "filing_date": self.filing_date,
            "sections": [s.to_dict() for s in self.sections],
            "chunks_count": len(self.chunks),
            "raw_text_length": len(self.raw_text),
            "cleaned_text_length": len(self.cleaned_text),
            "processing_time": self.processing_time,
            "metadata": self.metadata
        }
    
    def save_chunks(self, output_dir: Path) -> None:
        """Save chunks to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{Path(self.filename).stem}_chunks.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump([chunk.to_dict() for chunk in self.chunks], f, indent=2)


class DocumentPreprocessor:
    """
    Main document preprocessing class for SEC 10-K filings.
    Handles loading, cleaning, section extraction, and chunking.
    """
    
    # SEC 10-K Section patterns
    SECTION_PATTERNS = {
        "ITEM_1": {
            "patterns": [
                r"ITEM\s+1\.?\s*[-–—]?\s*BUSINESS",
                r"ITEM\s+1\.\s+BUSINESS",
            ],
            "name": "Business Description",
            "fraud_relevance": "medium"
        },
        "ITEM_1A": {
            "patterns": [
                r"ITEM\s+1A\.?\s*[-–—]?\s*RISK\s+FACTORS?",
                r"ITEM\s+1A\.\s+RISK\s+FACTORS?",
            ],
            "name": "Risk Factors",
            "fraud_relevance": "high"
        },
        "ITEM_1B": {
            "patterns": [
                r"ITEM\s+1B\.?\s*[-–—]?\s*UNRESOLVED\s+STAFF\s+COMMENTS?",
            ],
            "name": "Unresolved Staff Comments",
            "fraud_relevance": "high"
        },
        "ITEM_2": {
            "patterns": [
                r"ITEM\s+2\.?\s*[-–—]?\s*PROPERTIES",
            ],
            "name": "Properties",
            "fraud_relevance": "low"
        },
        "ITEM_3": {
            "patterns": [
                r"ITEM\s+3\.?\s*[-–—]?\s*LEGAL\s+PROCEEDINGS?",
            ],
            "name": "Legal Proceedings",
            "fraud_relevance": "high"
        },
        "ITEM_5": {
            "patterns": [
                r"ITEM\s+5\.?\s*[-–—]?\s*MARKET\s+FOR",
            ],
            "name": "Market for Common Equity",
            "fraud_relevance": "medium"
        },
        "ITEM_6": {
            "patterns": [
                r"ITEM\s+6\.?\s*[-–—]?\s*SELECTED\s+FINANCIAL\s+DATA",
            ],
            "name": "Selected Financial Data",
            "fraud_relevance": "high"
        },
        "ITEM_7": {
            "patterns": [
                r"ITEM\s+7\.?\s*[-–—]?\s*MANAGEMENT'?S?\s+DISCUSSION",
                r"MD&A",
            ],
            "name": "Management's Discussion and Analysis",
            "fraud_relevance": "critical"
        },
        "ITEM_7A": {
            "patterns": [
                r"ITEM\s+7A\.?\s*[-–—]?\s*QUANTITATIVE\s+AND\s+QUALITATIVE",
                r"ITEM\s+7A\.?\s*[-–—]?\s*MARKET\s+RISK",
            ],
            "name": "Market Risk Disclosures",
            "fraud_relevance": "medium"
        },
        "ITEM_8": {
            "patterns": [
                r"ITEM\s+8\.?\s*[-–—]?\s*FINANCIAL\s+STATEMENTS?",
            ],
            "name": "Financial Statements",
            "fraud_relevance": "critical"
        },
        "ITEM_9": {
            "patterns": [
                r"ITEM\s+9\.?\s*[-–—]?\s*CHANGES?\s+IN\s+AND\s+DISAGREEMENTS?",
            ],
            "name": "Changes in and Disagreements with Accountants",
            "fraud_relevance": "critical"
        },
        "ITEM_9A": {
            "patterns": [
                r"ITEM\s+9A\.?\s*[-–—]?\s*CONTROLS?\s+AND\s+PROCEDURES?",
            ],
            "name": "Controls and Procedures",
            "fraud_relevance": "critical"
        },
        "FOOTNOTES": {
            "patterns": [
                r"NOTES?\s+TO\s+(THE\s+)?CONSOLIDATED\s+FINANCIAL\s+STATEMENTS?",
                r"NOTES?\s+TO\s+FINANCIAL\s+STATEMENTS?",
            ],
            "name": "Notes to Financial Statements",
            "fraud_relevance": "critical"
        }
    }
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        config_path: Optional[str] = None
    ):
        """
        Initialize the document preprocessor.
        
        Args:
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Token overlap between chunks
            config_path: Path to configuration file
        """
        self.logger = get_preprocessing_logger()
        self.text_cleaner = TextCleaner()
        
        # Load configuration
        try:
            config = get_config(config_path)
            preproc_config = config.preprocessing_config
            self.chunk_size = preproc_config.get("chunk_size", chunk_size)
            self.chunk_overlap = preproc_config.get("chunk_overlap", chunk_overlap)
        except Exception:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        
        self.logger.info(f"Initialized DocumentPreprocessor with chunk_size={self.chunk_size}")
    
    def load_document(self, file_path: str) -> str:
        """
        Load document from file.
        
        Args:
            file_path: Path to the document file
        
        Returns:
            Raw document text
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension == ".txt":
            return self._load_txt(path)
        elif extension == ".pdf":
            return self._load_pdf(path)
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _load_txt(self, path: Path) -> str:
        """Load text file."""
        encodings = ['utf-8', 'latin-1', 'cp1252', 'ascii']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                self.logger.debug(f"Loaded {path} with {encoding} encoding")
                return content
            except UnicodeDecodeError:
                continue
        
        raise ValueError(f"Could not decode file with any supported encoding: {path}")
    
    def _load_pdf(self, path: Path) -> str:
        """Load PDF file using pdfplumber."""
        try:
            import pdfplumber
            
            text_parts = []
            with pdfplumber.open(path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text()
                    if text:
                        text_parts.append(f"[PAGE {page_num}]\n{text}")
            
            return "\n\n".join(text_parts)
        
        except ImportError:
            self.logger.warning("pdfplumber not installed. Trying pypdf...")
            return self._load_pdf_pypdf(path)
    
    def _load_pdf_pypdf(self, path: Path) -> str:
        """Fallback PDF loading with pypdf."""
        try:
            from pypdf import PdfReader
            
            reader = PdfReader(path)
            text_parts = []
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    text_parts.append(f"[PAGE {page_num}]\n{text}")
            
            return "\n\n".join(text_parts)
        
        except Exception as e:
            raise RuntimeError(f"Failed to load PDF: {e}")
    
    def remove_headers_footers(self, text: str) -> str:
        """
        Remove headers and footers from document text.
        
        Args:
            text: Raw document text
        
        Returns:
            Text with headers/footers removed
        """
        return self.text_cleaner.remove_headers_footers(text)
    
    def extract_relevant_sections(self, text: str) -> List[Section]:
        """
        Extract relevant sections from 10-K document.
        
        Args:
            text: Cleaned document text
        
        Returns:
            List of extracted sections
        """
        sections = []
        section_matches = []
        
        # Find all section matches
        for section_key, section_info in self.SECTION_PATTERNS.items():
            for pattern in section_info["patterns"]:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    section_matches.append({
                        "key": section_key,
                        "name": section_info["name"],
                        "fraud_relevance": section_info["fraud_relevance"],
                        "start": match.start(),
                        "end": match.end(),
                        "match_text": match.group()
                    })
        
        # Sort by position
        section_matches.sort(key=lambda x: x["start"])
        
        # Remove duplicates (keep first occurrence of each section)
        seen_sections = set()
        unique_matches = []
        for match in section_matches:
            if match["key"] not in seen_sections:
                seen_sections.add(match["key"])
                unique_matches.append(match)
        
        # Extract content between sections
        for i, match in enumerate(unique_matches):
            # Determine end position (start of next section or end of document)
            if i + 1 < len(unique_matches):
                end_pos = unique_matches[i + 1]["start"]
            else:
                end_pos = len(text)
            
            content = text[match["start"]:end_pos]
            
            section = Section(
                name=match["name"],
                section_type=match["key"],
                content=content,
                start_position=match["start"],
                end_position=end_pos,
                fraud_relevance=match["fraud_relevance"]
            )
            sections.append(section)
        
        self.logger.info(f"Extracted {len(sections)} sections from document")
        return sections
    
    def segment_into_chunks(
        self,
        text: str,
        section_type: str = "UNKNOWN",
        section_name: str = "Unknown",
        base_position: int = 0
    ) -> List[TextChunk]:
        """
        Segment text into overlapping chunks.
        
        Args:
            text: Text to segment
            section_type: Type of section
            section_name: Name of section
            base_position: Starting position in original document
        
        Returns:
            List of TextChunk objects
        """
        chunks = []
        words = text.split()
        
        if not words:
            return chunks
        
        # Approximate tokens as words (rough estimate)
        current_position = 0
        chunk_index = 0
        
        while current_position < len(words):
            # Get chunk words
            chunk_words = words[current_position:current_position + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            # Calculate positions
            start_char = text.find(chunk_words[0]) if chunk_words else 0
            end_char = start_char + len(chunk_text)
            
            # Generate chunk ID
            chunk_id = self._generate_chunk_id(chunk_text, section_type, chunk_index)
            
            chunk = TextChunk(
                id=chunk_id,
                text=chunk_text,
                section_type=section_type,
                section_name=section_name,
                page_number=self._extract_page_number(chunk_text),
                chunk_index=chunk_index,
                start_position=base_position + current_position,
                end_position=base_position + current_position + len(chunk_words),
                word_count=len(chunk_words),
                metadata={
                    "estimated_tokens": len(chunk_words) * 1.3  # Rough estimate
                }
            )
            chunks.append(chunk)
            
            # Move to next chunk with overlap
            current_position += self.chunk_size - self.chunk_overlap
            chunk_index += 1
            
            # Prevent infinite loop for small texts
            if current_position >= len(words):
                break
        
        return chunks
    
    def _generate_chunk_id(
        self,
        text: str,
        section_type: str,
        chunk_index: int
    ) -> str:
        """Generate unique chunk ID."""
        content_hash = hashlib.md5(text[:100].encode()).hexdigest()[:8]
        return f"{section_type}_{chunk_index}_{content_hash}"
    
    def _extract_page_number(self, text: str) -> Optional[int]:
        """Extract page number from text if present."""
        page_pattern = r"\[PAGE\s+(\d+)\]"
        match = re.search(page_pattern, text)
        if match:
            return int(match.group(1))
        return None
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for processing.
        
        Args:
            text: Input text
        
        Returns:
            Normalized text
        """
        result = self.text_cleaner.clean(text)
        return result.cleaned_text
    
    def extract_company_info(self, text: str) -> Dict:
        """
        Extract company information from the filing.
        
        Args:
            text: Document text (first portion)
        
        Returns:
            Dictionary with company info
        """
        info = {
            "company_name": None,
            "fiscal_year": None,
            "filing_date": None,
            "cik": None
        }
        
        # Try to extract company name
        company_patterns = [
            r"(?:^|\n)\s*([A-Z][A-Z\s\.,&]+(?:INC|CORP|LLC|LTD|CO|COMPANY)?\.?)\s*$",
            r"EXACT NAME OF REGISTRANT[^:]*:\s*([^\n]+)",
        ]
        
        for pattern in company_patterns:
            match = re.search(pattern, text[:5000], re.MULTILINE | re.IGNORECASE)
            if match:
                info["company_name"] = match.group(1).strip()
                break
        
        # Try to extract fiscal year
        fy_pattern = r"(?:fiscal|for the)\s+year\s+ended?\s+(\w+\s+\d{1,2},?\s+\d{4})"
        match = re.search(fy_pattern, text[:10000], re.IGNORECASE)
        if match:
            info["fiscal_year"] = match.group(1)
        
        # Try to extract CIK
        cik_pattern = r"CIK[:\s]+(\d+)"
        match = re.search(cik_pattern, text[:5000], re.IGNORECASE)
        if match:
            info["cik"] = match.group(1)
        
        return info
    
    def process(self, file_path: str) -> ProcessedDocument:
        """
        Main processing method - processes entire document.
        
        Args:
            file_path: Path to document file
        
        Returns:
            ProcessedDocument object
        """
        start_time = datetime.now()
        self.logger.log_module_start("DocumentPreprocessor")
        
        # Load document
        self.logger.info(f"Loading document: {file_path}")
        raw_text = self.load_document(file_path)
        
        # Clean text
        self.logger.info("Cleaning document text...")
        cleaning_result = self.text_cleaner.clean(raw_text)
        cleaned_text = cleaning_result.cleaned_text
        
        # Extract company info
        company_info = self.extract_company_info(cleaned_text)
        
        # Extract sections
        self.logger.info("Extracting sections...")
        sections = self.extract_relevant_sections(cleaned_text)
        
        # Chunk all sections
        self.logger.info("Segmenting into chunks...")
        all_chunks = []
        for section in sections:
            section_chunks = self.segment_into_chunks(
                section.content,
                section_type=section.section_type,
                section_name=section.name,
                base_position=section.start_position
            )
            all_chunks.extend(section_chunks)
        
        # If no sections found, chunk the entire document
        if not all_chunks:
            self.logger.warning("No sections found, chunking entire document")
            all_chunks = self.segment_into_chunks(
                cleaned_text,
                section_type="FULL_DOCUMENT",
                section_name="Full Document"
            )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        self.logger.log_module_end("DocumentPreprocessor", processing_time)
        
        # Create processed document
        processed_doc = ProcessedDocument(
            filename=Path(file_path).name,
            file_path=str(file_path),
            company_name=company_info.get("company_name"),
            fiscal_year=company_info.get("fiscal_year"),
            filing_date=company_info.get("filing_date"),
            sections=sections,
            chunks=all_chunks,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            processing_time=processing_time,
            metadata={
                "cik": company_info.get("cik"),
                "original_length": cleaning_result.original_length,
                "cleaned_length": cleaning_result.cleaned_length,
                "cleaning_changes": cleaning_result.changes_made,
                "section_count": len(sections),
                "chunk_count": len(all_chunks)
            }
        )
        
        self.logger.info(
            f"Processed document: {len(sections)} sections, {len(all_chunks)} chunks"
        )
        
        return processed_doc
    
    def process_batch(
        self,
        file_paths: List[str]
    ) -> Iterator[ProcessedDocument]:
        """
        Process multiple documents.
        
        Args:
            file_paths: List of file paths
        
        Yields:
            ProcessedDocument objects
        """
        for file_path in file_paths:
            try:
                yield self.process(file_path)
            except Exception as e:
                self.logger.error(f"Failed to process {file_path}: {e}")
                continue
