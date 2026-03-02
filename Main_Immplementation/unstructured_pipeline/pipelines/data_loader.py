
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
        limit: Optional[int] = None,
        specific_file: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Load all documents from the data directory or a specific file
        
        Args:
            file_pattern: Glob pattern for files to load (ignored if specific_file is provided)
            limit: Maximum number of files to load (None for all)
            specific_file: Specific filename to load (e.g., "full-submission.txt")
            
        Returns:
            List of document dictionaries
        """
        try:
            self.logger.info(f"Loading documents from {self.data_dir}")
            
            # Get all matching files
            if specific_file:
                # Load only the specific file
                file_path = self.data_dir / specific_file
                if not file_path.exists():
                    self.logger.error(f"Specific file not found: {file_path}")
                    raise DataIngestionError(f"File not found: {specific_file}")
                files = [file_path]
                self.logger.info(f"Loading specific file: {specific_file}")
            else:
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
            
            # Parse filename to extract metadata
            metadata = self._parse_filename(file_path.stem)
            
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
    
    def _parse_filename(self, filename: str) -> Dict[str, Any]:
        """
        Parse metadata from filename
        Examples: 
        - NonFraud_1961_20200330_1376.txt
        - Unknown_full-submission_163
        """
        metadata = {
            'label': 'unknown',
            'company_id': None,
            'date': None,
            'submission_id': None
        }
        
        # Pattern for fraud label files
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
