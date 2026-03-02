# File: pipelines/chunking.py
"""
Text chunking utilities for document processing
"""
from typing import List, Dict, Any
import re
from pathlib import Path

from utils import Config, Logger


class TextChunker:
    """Handles text chunking for document processing"""
    
    def __init__(
        self,
        chunk_size: int = Config.CHUNK_SIZE,
        chunk_overlap: int = Config.CHUNK_OVERLAP
    ):
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            doc_id: Document identifier
            
        Returns:
            List of chunk dictionaries with metadata
        """
        # Clean the text
        text = self._clean_text(text)
        
        # Split into sentences for better chunking
        sentences = self._split_sentences(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk size, save current chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                    'doc_id': doc_id,
                    'chunk_index': chunk_id,
                    'length': current_length
                })
                
                # Keep overlap
                overlap_words = ' '.join(current_chunk).split()[-self.chunk_overlap:]
                current_chunk = [' '.join(overlap_words)]
                current_length = len(overlap_words)
                chunk_id += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append({
                'text': chunk_text,
                'chunk_id': f"{doc_id}_chunk_{chunk_id}",
                'doc_id': doc_id,
                'chunk_index': chunk_id,
                'length': current_length
            })
        
        self.logger.debug(f"Created {len(chunks)} chunks from document {doc_id}")
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.,;:!?()-]', '', text)
        return text.strip()
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (can be improved with spaCy)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

