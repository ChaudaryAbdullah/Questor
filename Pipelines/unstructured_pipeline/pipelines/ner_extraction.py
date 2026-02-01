# File: pipelines/ner_extraction_optimized.py
"""
Memory-Optimized Named Entity Recognition and Relationship Extraction
Handles large documents by processing in chunks
"""
import spacy
import subprocess
import sys
from typing import List, Dict, Any, Optional
from collections import defaultdict
import re
import gc

from utils import Config, Logger
from utils.exceptions import NERExtractionError


class NERExtractorOptimized:
    """
    Memory-optimized entity extraction
    
    Key improvements:
    1. Process documents in smaller chunks
    2. Stream processing instead of loading all at once
    3. Lazy model loading
    4. Explicit memory cleanup
    """
    
    FINANCIAL_TERMS = {
        'special purpose entity', 'spe', 'derivative', 'goodwill', 'restructuring',
        'restatement', 'write-off', 'write-down', 'impairment', 'round-trip',
        'revenue recognition', 'accounts receivable', 'liability', 'subsidiary',
        'joint venture', 'related party', 'off-balance sheet', 'contingency'
    }
    
    FRAUD_INDICATORS = {
        'fictitious', 'fabricated', 'manipulation', 'misstatement', 'overstatement',
        'understatement', 'concealment', 'material weakness', 'restatement'
    }
    
    def __init__(self, model_name: str = Config.SPACY_MODEL, chunk_size: int = 500000):
        """
        Initialize NER extractor
        
        Args:
            model_name: spaCy model name
            chunk_size: Maximum characters per chunk (default 500K)
        """
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.nlp = None
        # Model is loaded lazily
    
    def _ensure_model_loaded(self):
        """Lazy load spaCy model only when needed"""
        if self.nlp is not None:
            return
        
        try:
            self.logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            # Set reasonable max_length
            self.nlp.max_length = 1000000
            self.logger.info("spaCy model loaded successfully")
        except OSError:
            self.logger.info(f"Downloading spaCy model: {self.model_name}")
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", self.model_name])
                self.nlp = spacy.load(self.model_name)
                self.nlp.max_length = 1000000
                self.logger.info("spaCy model downloaded and loaded")
            except Exception as e:
                raise NERExtractionError(f"Failed to load model: {str(e)}")
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from text with memory optimization
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entities grouped by type
        """
        self._ensure_model_loaded()
        
        try:
            entities = defaultdict(list)
            
            # Process in chunks if text is large
            if len(text) > self.chunk_size:
                chunks = self._split_text_smart(text, self.chunk_size)
                self.logger.debug(f"Processing {len(chunks)} chunks for entity extraction")
                
                offset = 0
                for chunk in chunks:
                    try:
                        chunk_entities = self._extract_from_chunk(chunk, offset)
                        
                        # Merge entities
                        for entity_type, entity_list in chunk_entities.items():
                            entities[entity_type].extend(entity_list)
                        
                        offset += len(chunk)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process chunk at offset {offset}: {str(e)}")
                        continue
                    
                    # Cleanup after each chunk
                    gc.collect()
            else:
                entities = self._extract_from_chunk(text, 0)
            
            # Deduplicate entities
            entities = self._deduplicate_entities(entities)
            
            return dict(entities)
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {str(e)}")
            raise NERExtractionError(f"Entity extraction failed: {str(e)}")
    
    def _extract_from_chunk(self, text: str, offset: int = 0) -> Dict[str, List[Dict]]:
        """Extract entities from a single chunk"""
        entities = defaultdict(list)
        
        # Process with spaCy
        doc = self.nlp(text)
        
        for ent in doc.ents:
            entity_data = {
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char + offset,
                'end': ent.end_char + offset
            }
            entities[ent.label_].append(entity_data)
        
        # Extract financial terms
        financial_entities = self._extract_financial_terms(text, offset)
        if financial_entities:
            entities['FINANCIAL_TERM'].extend(financial_entities)
        
        # Extract fraud indicators
        fraud_entities = self._extract_fraud_indicators(text, offset)
        if fraud_entities:
            entities['FRAUD_INDICATOR'].extend(fraud_entities)
        
        # Extract monetary amounts
        monetary_entities = self._extract_monetary_amounts(text, offset)
        if monetary_entities:
            entities['MONEY'].extend(monetary_entities)
        
        return entities
    
    def extract_relationships(
        self,
        text: str,
        entities: Optional[Dict[str, List[Dict]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships with memory optimization
        
        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
            
        Returns:
            List of relationship dictionaries
        """
        self._ensure_model_loaded()
        
        try:
            relationships = []
            
            # Process in chunks if text is large
            if len(text) > self.chunk_size:
                chunks = self._split_text_smart(text, self.chunk_size)
                
                for chunk in chunks:
                    try:
                        chunk_rels = self._extract_relationships_from_chunk(chunk, entities)
                        relationships.extend(chunk_rels)
                    except Exception as e:
                        self.logger.warning(f"Failed to extract relationships: {str(e)}")
                        continue
                    
                    gc.collect()
            else:
                relationships = self._extract_relationships_from_chunk(text, entities)
            
            # Deduplicate relationships
            relationships = self._deduplicate_relationships(relationships)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Relationship extraction failed: {str(e)}")
            raise NERExtractionError(f"Relationship extraction failed: {str(e)}")
    
    def _extract_relationships_from_chunk(
        self,
        text: str,
        entities: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Extract relationships from a single chunk"""
        relationships = []
        doc = self.nlp(text)
        
        # Extract subject-verb-object relationships
        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass'):
                subject = token.text
                verb = token.head.text
                
                for child in token.head.children:
                    if child.dep_ in ('dobj', 'pobj', 'attr'):
                        obj = child.text
                        
                        relationship = {
                            'subject': subject,
                            'predicate': verb,
                            'object': obj,
                            'relation_type': self._classify_relationship(verb)
                        }
                        relationships.append(relationship)
        
        # Extract fraud-related relationships
        fraud_rels = self._extract_fraud_relationships(doc, entities or {})
        relationships.extend(fraud_rels)
        
        return relationships
    
    def _split_text_smart(self, text: str, max_length: int) -> List[str]:
        """
        Split text into chunks while preserving sentence boundaries
        
        Args:
            text: Text to split
            max_length: Maximum length per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split by paragraphs
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            para_len = len(para)
            
            if current_length + para_len <= max_length:
                current_chunk.append(para)
                current_length += para_len
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append('\n\n'.join(current_chunk))
                
                # Handle oversized paragraph
                if para_len > max_length:
                    # Split by sentences
                    sentences = para.split('. ')
                    temp_chunk = []
                    temp_length = 0
                    
                    for sent in sentences:
                        sent_len = len(sent)
                        if temp_length + sent_len <= max_length:
                            temp_chunk.append(sent)
                            temp_length += sent_len
                        else:
                            if temp_chunk:
                                chunks.append('. '.join(temp_chunk) + '.')
                            temp_chunk = [sent]
                            temp_length = sent_len
                    
                    if temp_chunk:
                        current_chunk = ['. '.join(temp_chunk) + '.']
                        current_length = sum(len(s) for s in temp_chunk)
                else:
                    current_chunk = [para]
                    current_length = para_len
        
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _deduplicate_entities(self, entities: Dict) -> Dict:
        """Remove duplicate entities based on text"""
        deduplicated = {}
        for entity_type, entity_list in entities.items():
            seen = set()
            unique_entities = []
            for entity in entity_list:
                key = (entity['text'].lower(), entity['label'])
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            deduplicated[entity_type] = unique_entities
        return deduplicated
    
    def _deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """Remove duplicate relationships"""
        seen = set()
        unique_rels = []
        for rel in relationships:
            key = (
                rel['subject'].lower(),
                rel['predicate'].lower(),
                rel['object'].lower()
            )
            if key not in seen:
                seen.add(key)
                unique_rels.append(rel)
        return unique_rels
    
    def _extract_financial_terms(self, text: str, offset: int = 0) -> List[Dict[str, Any]]:
        """Extract financial terminology"""
        text_lower = text.lower()
        entities = []
        
        for term in self.FINANCIAL_TERMS:
            pos = 0
            while True:
                pos = text_lower.find(term, pos)
                if pos == -1:
                    break
                entities.append({
                    'text': term,
                    'label': 'FINANCIAL_TERM',
                    'start': pos + offset,
                    'end': pos + len(term) + offset
                })
                pos += len(term)
        
        return entities
    
    def _extract_fraud_indicators(self, text: str, offset: int = 0) -> List[Dict[str, Any]]:
        """Extract fraud indicator terms"""
        text_lower = text.lower()
        entities = []
        
        for indicator in self.FRAUD_INDICATORS:
            pos = 0
            while True:
                pos = text_lower.find(indicator, pos)
                if pos == -1:
                    break
                entities.append({
                    'text': indicator,
                    'label': 'FRAUD_INDICATOR',
                    'start': pos + offset,
                    'end': pos + len(indicator) + offset
                })
                pos += len(indicator)
        
        return entities
    
    def _extract_monetary_amounts(self, text: str, offset: int = 0) -> List[Dict[str, Any]]:
        """Extract monetary amounts using regex"""
        pattern = r'\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|B|K))?'
        entities = []
        
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                'text': match.group(),
                'label': 'MONEY',
                'start': match.start() + offset,
                'end': match.end() + offset
            })
        
        return entities
    
    def _classify_relationship(self, verb: str) -> str:
        """Classify relationship type based on verb"""
        verb_lower = verb.lower()
        
        relationship_types = {
            'is': 'IS',
            'has': 'HAS',
            'owns': 'OWNS',
            'manages': 'MANAGES',
            'controls': 'CONTROLS',
            'reports': 'REPORTS_TO',
            'created': 'CREATED',
            'transferred': 'TRANSFERRED',
            'hid': 'CONCEALED',
            'concealed': 'CONCEALED'
        }
        
        return relationship_types.get(verb_lower, 'RELATED_TO')
    
    def _extract_fraud_relationships(self, doc, entities: Dict) -> List[Dict[str, Any]]:
        """Extract specific fraud-related relationships"""
        relationships = []
        
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            if any(indicator in sent_text for indicator in ['hid', 'concealed', 'transferred']):
                orgs = [ent.text for ent in sent.ents if ent.label_ == 'ORG']
                money = [ent.text for ent in sent.ents if ent.label_ == 'MONEY']
                
                if orgs and money:
                    relationships.append({
                        'subject': orgs[0],
                        'predicate': 'CONCEALED',
                        'object': money[0],
                        'relation_type': 'FRAUD_ACTION'
                    })
        
        return relationships
    
    def unload_model(self):
        """Explicitly unload the spaCy model to free memory"""
        if self.nlp is not None:
            del self.nlp
            self.nlp = None
            gc.collect()
            self.logger.info("spaCy model unloaded from memory")