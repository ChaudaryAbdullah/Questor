"""
CIK Extractor Utility
Extracts and normalizes CIK numbers from JSON input files
"""
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import logging
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from utils import Logger
except ImportError:
    # Fallback to basic logging if utils not available
    class Logger:
        @staticmethod
        def get_logger(name):
            return logging.getLogger(name)


class CIKExtractor:
    """Extract and normalize CIK numbers from input files"""
    
    def __init__(self):
        self.logger = Logger.get_logger(self.__class__.__name__)
    
    @staticmethod
    def normalize_cik(cik: str) -> str:
        """
        Normalize CIK by removing leading zeros
        
        Args:
            cik: CIK string (e.g., "0001040719")
            
        Returns:
            Normalized CIK without leading zeros (e.g., "1040719")
        """
        try:
            # Convert to int to remove leading zeros, then back to string
            return str(int(cik))
        except (ValueError, TypeError):
            # If conversion fails, return as-is
            return str(cik)
    
    def extract_cik_from_file(self, file_path: Path) -> Optional[str]:
        """
        Extract and normalize CIK from a single JSON file
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Normalized CIK string or None if not found
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract CIK field
            cik = data.get('cik')
            
            if cik:
                normalized_cik = self.normalize_cik(cik)
                self.logger.debug(f"Extracted CIK from {file_path.name}: {cik} -> {normalized_cik}")
                return normalized_cik
            else:
                self.logger.warning(f"No 'cik' field found in {file_path.name}")
                return None
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON from {file_path.name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to extract CIK from {file_path.name}: {e}")
            return None
    
    def extract_from_directory(
        self,
        directory: Path,
        file_pattern: str = "*.json"
    ) -> List[Tuple[str, str]]:
        """
        Extract CIKs from all JSON files in a directory
        
        Args:
            directory: Directory containing JSON files
            file_pattern: Glob pattern for files to process
            
        Returns:
            List of (filename, normalized_cik) tuples
        """
        results = []
        
        if not directory.exists():
            self.logger.error(f"Directory does not exist: {directory}")
            return results
        
        json_files = list(directory.glob(file_pattern))
        self.logger.info(f"Found {len(json_files)} JSON files in {directory}")
        
        for file_path in json_files:
            cik = self.extract_cik_from_file(file_path)
            if cik:
                results.append((file_path.name, cik))
        
        self.logger.info(f"Extracted {len(results)} CIKs from {len(json_files)} files")
        return results
    
    def extract_multiple_files(
        self,
        file_paths: List[Path]
    ) -> List[Tuple[str, str]]:
        """
        Extract CIKs from a list of files
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of (filename, normalized_cik) tuples
        """
        results = []
        
        for file_path in file_paths:
            cik = self.extract_cik_from_file(file_path)
            if cik:
                results.append((file_path.name, cik))
        
        return results
    
    def get_cik_file_mapping(
        self,
        directory: Path,
        file_pattern: str = "*.json"
    ) -> Dict[str, str]:
        """
        Get a mapping of CIK to filename
        
        Args:
            directory: Directory containing JSON files
            file_pattern: Glob pattern for files
            
        Returns:
            Dictionary mapping normalized_cik -> filename
        """
        results = self.extract_from_directory(directory, file_pattern)
        return {cik: filename for filename, cik in results}


def extract_cik_from_json(file_path: str) -> Optional[str]:
    """
    Convenience function to extract CIK from a JSON file
    
    Args:
        file_path: Path to JSON file (string or Path)
        
    Returns:
        Normalized CIK or None
    """
    extractor = CIKExtractor()
    return extractor.extract_cik_from_file(Path(file_path))


def main():
    """Test the CIK extractor"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract CIK numbers from JSON files')
    parser.add_argument('--file', type=str, help='Single JSON file to process')
    parser.add_argument('--directory', type=str, help='Directory containing JSON files')
    parser.add_argument('--pattern', type=str, default='*.json', help='File pattern (default: *.json)')
    
    args = parser.parse_args()
    
    extractor = CIKExtractor()
    
    if args.file:
        cik = extractor.extract_cik_from_file(Path(args.file))
        if cik:
            print(f"✓ CIK: {cik}")
        else:
            print("✗ Failed to extract CIK")
    
    elif args.directory:
        results = extractor.extract_from_directory(Path(args.directory), args.pattern)
        print(f"\nExtracted {len(results)} CIKs:")
        for filename, cik in results:
            print(f"  {filename}: {cik}")
    
    else:
        print("Error: Specify either --file or --directory")


if __name__ == "__main__":
    main()
