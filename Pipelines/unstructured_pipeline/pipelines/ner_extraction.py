# File: pipelines/ner_extraction.py
"""
Named Entity Recognition and Relationship Extraction
"""
import spacy
import subprocess
import sys
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import re

from utils import Config, Logger
from utils.exceptions import NERExtractionError


class NERExtractor:
    """Extract entities and relationships from text using spaCy"""
    
    # Financial terms and fraud indicators
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
    
    def __init__(self, model_name: str = Config.SPACY_MODEL):
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.model_name = model_name
        self.nlp = None
        self._load_model()
    
    def _load_model(self):
        """Load spaCy NER model"""
        try:
            self.logger.info(f"Loading spaCy model: {self.model_name}")
            self.nlp = spacy.load(self.model_name)
            # Increase max_length to handle longer texts
            self.nlp.max_length = 3000000  # 3 million characters
            self.logger.info("spaCy model loaded successfully")
        except OSError as e:
            self.logger.warning(f"spaCy model not found: {str(e)}")
            self.logger.info(f"Attempting to download spaCy model: {self.model_name}")
            try:
                subprocess.check_call([sys.executable, "-m", "spacy", "download", self.model_name])
                self.nlp = spacy.load(self.model_name)
                # Increase max_length to handle longer texts
                self.nlp.max_length = 3000000  # 3 million characters
                self.logger.info("spaCy model downloaded and loaded successfully")
            except Exception as download_error:
                self.logger.error(f"Failed to download spaCy model: {str(download_error)}")
                raise NERExtractionError(f"Model loading failed: {str(download_error)}")
        except Exception as e:
            self.logger.error(f"Failed to load spaCy model: {str(e)}")
            raise NERExtractionError(f"Model loading failed: {str(e)}")
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entities grouped by type
        """
        try:
            entities = defaultdict(list)
            
            # Handle very long texts by processing in chunks
            max_chunk_length = 900000  # Process 900k char chunks to be safe
            if len(text) > max_chunk_length:
                self.logger.info(f"Text length {len(text)} exceeds chunk limit, processing in chunks...")
                chunks = self._split_text_into_chunks(text, max_chunk_length)
                
                for i, chunk in enumerate(chunks):
                    try:
                        doc = self.nlp(chunk)
                        
                        for ent in doc.ents:
                            entity_data = {
                                'text': ent.text,
                                'label': ent.label_,
                                'start': ent.start_char,
                                'end': ent.end_char
                            }
                            entities[ent.label_].append(entity_data)
                    except Exception as chunk_error:
                        self.logger.warning(f"Failed to process chunk {i+1}: {str(chunk_error)}")
                        continue
            else:
                doc = self.nlp(text)
                
                for ent in doc.ents:
                    entity_data = {
                        'text': ent.text,
                        'label': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    }
                    entities[ent.label_].append(entity_data)
            
            # Extract financial terms
            financial_entities = self._extract_financial_terms(text)
            if financial_entities:
                entities['FINANCIAL_TERM'] = financial_entities
            
            # Extract fraud indicators
            fraud_entities = self._extract_fraud_indicators(text)
            if fraud_entities:
                entities['FRAUD_INDICATOR'] = fraud_entities
            
            # Extract monetary amounts
            monetary_entities = self._extract_monetary_amounts(text)
            if monetary_entities:
                entities['MONEY'].extend(monetary_entities)
            
            return dict(entities)
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {str(e)}")
            raise NERExtractionError(f"Entity extraction failed: {str(e)}")
    
    def extract_relationships(
        self,
        text: str,
        entities: Optional[Dict[str, List[Dict]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities
        
        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
            
        Returns:
            List of relationship dictionaries
        """
        try:
            relationships = []
            
            # Handle very long texts by processing in chunks
            max_chunk_length = 900000  # Process 900k char chunks to be safe
            if len(text) > max_chunk_length:
                self.logger.info(f"Text length {len(text)} exceeds chunk limit for relationships, processing in chunks...")
                chunks = self._split_text_into_chunks(text, max_chunk_length)
                
                for i, chunk in enumerate(chunks):
                    try:
                        doc = self.nlp(chunk)
                        
                        # Extract subject-verb-object relationships
                        for token in doc:
                            if token.dep_ in ('nsubj', 'nsubjpass'):
                                subject = token.text
                                verb = token.head.text
                                
                                # Find object
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
                        
                        # Extract specific fraud-related relationships
                        fraud_relationships = self._extract_fraud_relationships(doc, entities or {})
                        relationships.extend(fraud_relationships)
                    except Exception as chunk_error:
                        self.logger.warning(f"Failed to process chunk {i+1} for relationships: {str(chunk_error)}")
                        continue
            else:
                doc = self.nlp(text)
                
                if entities is None:
                    entities = self.extract_entities(text)
                
                # Extract subject-verb-object relationships
                for token in doc:
                    if token.dep_ in ('nsubj', 'nsubjpass'):
                        subject = token.text
                        verb = token.head.text
                        
                        # Find object
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
                
                # Extract specific fraud-related relationships
                fraud_relationships = self._extract_fraud_relationships(doc, entities)
                relationships.extend(fraud_relationships)
            
            return relationships
            
        except Exception as e:
            self.logger.error(f"Relationship extraction failed: {str(e)}")
            raise NERExtractionError(f"Relationship extraction failed: {str(e)}")
    
    def _extract_financial_terms(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial terminology"""
        text_lower = text.lower()
        entities = []
        
        for term in self.FINANCIAL_TERMS:
            if term in text_lower:
                start = text_lower.find(term)
                entities.append({
                    'text': term,
                    'label': 'FINANCIAL_TERM',
                    'start': start,
                    'end': start + len(term)
                })
        
        return entities
    
    def _extract_fraud_indicators(self, text: str) -> List[Dict[str, Any]]:
        """Extract fraud indicator terms"""
        text_lower = text.lower()
        entities = []
        
        for indicator in self.FRAUD_INDICATORS:
            if indicator in text_lower:
                start = text_lower.find(indicator)
                entities.append({
                    'text': indicator,
                    'label': 'FRAUD_INDICATOR',
                    'start': start,
                    'end': start + len(indicator)
                })
        
        return entities
    
    def _extract_monetary_amounts(self, text: str) -> List[Dict[str, Any]]:
        """Extract monetary amounts using regex"""
        # Pattern for amounts like $50 million, $1.2 billion, etc.
        pattern = r'\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|thousand|M|B|K))?'
        
        entities = []
        for match in re.finditer(pattern, text, re.IGNORECASE):
            entities.append({
                'text': match.group(),
                'label': 'MONEY',
                'start': match.start(),
                'end': match.end()
            })
        
        return entities
    
    def _classify_relationship(self, verb: str) -> str:
        """Classify the type of relationship based on verb"""
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
    
    def _extract_fraud_relationships(
        self,
        doc,
        entities: Dict[str, List[Dict]]
    ) -> List[Dict[str, Any]]:
        """Extract specific fraud-related relationships"""
        relationships = []
        
        # Look for patterns like "Company X hid $Y in debt"
        for sent in doc.sents:
            sent_text = sent.text.lower()
            
            if any(indicator in sent_text for indicator in ['hid', 'concealed', 'transferred']):
                # Try to extract company and amount
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
    
    def _split_text_into_chunks(self, text: str, max_length: int) -> List[str]:
        """
        Split text into chunks of approximately max_length characters
        while trying to preserve sentence boundaries
        
        Args:
            text: Text to split
            max_length: Maximum length per chunk
            
        Returns:
            List of text chunks
        """
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            # If paragraph itself is too long, split by sentences
            if len(paragraph) > max_length:
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_length:
                        current_chunk += sentence + ". "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
            else:
                if len(current_chunk) + len(paragraph) <= max_length:
                    current_chunk += paragraph + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = paragraph + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def extract_document_entities(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract entities from multiple documents
        
        Args:
            documents: List of document dictionaries with 'content' key
            
        Returns:
            List of documents with added 'entities' and 'relationships'
        """
        self.logger.info(f"Extracting entities from {len(documents)} documents...")
        
        processed_docs = []
        
        for doc in documents:
            try:
                content = doc.get('content', '')
                
                # Extract entities
                entities = self.extract_entities(content)
                
                # Extract relationships
                relationships = self.extract_relationships(content, entities)
                
                # Add to document
                doc['entities'] = entities
                doc['relationships'] = relationships
                
                processed_docs.append(doc)
                
            except Exception as e:
                self.logger.warning(f"Failed to extract from doc {doc.get('doc_id')}: {str(e)}")
                continue
        
        self.logger.info(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs

