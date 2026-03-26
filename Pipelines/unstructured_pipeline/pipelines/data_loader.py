
# File: pipelines/data_loader.py
"""
Data loading utilities for unstructured documents
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import re
from tqdm import tqdm

from utils import Config, Logger
from utils.exceptions import DataIngestionError


class DataLoader:
    """Loads and preprocesses unstructured documents"""
    
    def __init__(self, data_dir: Path = Config.DATA_DIR):
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.data_dir = data_dir
    
    def load_documents(
        self,
        file_pattern: str = "*.txt",
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load all documents from the data directory
        
        Args:
            file_pattern: Glob pattern for files to load
            limit: Maximum number of files to load (None for all)
            
        Returns:
            List of document dictionaries
        """
        try:
            self.logger.info(f"Loading documents from {self.data_dir}")
            
            # Get all matching files
            files = list(self.data_dir.glob(file_pattern))
            
            if limit:
                files = files[:limit]
            
            self.logger.info(f"Found {len(files)} documents to process")
            
            documents = []
            
            for file_path in tqdm(files, desc="Loading documents"):
                doc = self._load_single_document(file_path)
                if doc:
                    documents.append(doc)
            
            self.logger.info(f"Successfully loaded {len(documents)} documents")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to load documents: {str(e)}")
            raise DataIngestionError(f"Document loading failed: {str(e)}")
    
    def _load_single_document(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Load a single document file"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse filename to extract metadata; pass content for fallback extraction
            metadata = self._parse_filename(file_path.stem, content)
            
            return {
                'doc_id': file_path.stem,
                'content': content,
                'file_path': str(file_path),
                'file_name': file_path.name,
                **metadata
            }
            
        except Exception as e:
            self.logger.warning(f"Failed to load {file_path.name}: {str(e)}")
            return None
    
    def _parse_filename(self, filename: str, content: str = '') -> Dict[str, Any]:
        """
        Parse metadata from filename
        Examples: 
        - NonFraud_1961_20200330_1376.txt
        - Unknown_full-submission_163
        
        For Unknown_full documents, CIK and date are extracted from the
        embedded "Source File:" header in the file content.
        """
        metadata = {
            'label': 'fraud',
            'company_id': None,
            'date': None,
            'submission_id': None
        }
        
        # Pattern for labeled files: NonFraud or Fraud
        fraud_pattern = r'(NonFraud|Fraud)_(\d+)_(\d{8})_(\d+)'
        match = re.match(fraud_pattern, filename)
        
        if match:
            metadata['label'] = match.group(1).lower()
            metadata['company_id'] = match.group(2)
            metadata['date'] = match.group(3)
            metadata['submission_id'] = match.group(4)
        else:
            # Pattern for unknown submissions
            unknown_pattern = r'Unknown_full-submission_(\d+)'
            match = re.match(unknown_pattern, filename)
            if match:
                metadata['submission_id'] = match.group(1)
            
            # Try to extract CIK and date from the embedded file header
            # Expected header line:
            # Source File: .../sec-edgar-filings/<CIK>/<form>/<accession>/full-submission.txt
            # Accession number format: XXXXXXXXXX-YY-NNNNNN  (YY = 2-digit year)
            if content:
                # Extract CIK from the EDGAR path
                cik_match = re.search(
                    r'sec-edgar-filings[/\\](\d+)[/\\]',
                    content[:500]  # Only scan the header
                )
                if cik_match:
                    metadata['company_id'] = cik_match.group(1).lstrip('0') or cik_match.group(1)
                
                # Extract date from accession number (XXXXXXXXXX-YY-NNNNNN)
                # YY is the 2-digit filing year; we map it to a full 8-digit date YYYYMMDD
                accession_match = re.search(
                    r'(\d{10})-(\d{2})-(\d{6})',
                    content[:500]
                )
                if accession_match:
                    year_short = int(accession_match.group(2))
                    # SEC EDGAR 2-digit years: <=30 -> 2000s, else 1900s
                    year_full = 2000 + year_short if year_short <= 30 else 1900 + year_short
                    metadata['date'] = f"{year_full}0101"  # Use Jan 1 as approximate date
        
        return metadata
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        files = list(self.data_dir.glob("*.txt"))
        
        fraud_count = sum(1 for f in files if 'NonFraud' in f.name or 'Fraud' in f.name)
        unknown_count = sum(1 for f in files if 'Unknown' in f.name)
        
        return {
            'total_documents': len(files),
            'fraud_labeled': fraud_count,
            'unknown': unknown_count,
            'data_directory': str(self.data_dir)
        }
