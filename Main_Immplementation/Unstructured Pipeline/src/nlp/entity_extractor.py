"""
Main Entity Extractor Module
Coordinates entity extraction from SEC filings using LLM (Groq) and spaCy fallback.
"""

import json
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import yaml
from pathlib import Path

from ..utils.logger import get_nlp_logger
from ..utils.config_manager import get_config


@dataclass
class ExtractedEntity:
    """Represents an extracted entity."""
    id: str
    text: str
    entity_type: str
    sub_type: Optional[str] = None
    confidence: float = 0.0
    start_position: int = 0
    end_position: int = 0
    section: str = ""
    page_number: Optional[int] = None
    attributes: Dict = field(default_factory=dict)
    fraud_indicators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExtractedRelationship:
    """Represents an extracted relationship between entities."""
    id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str
    confidence: float = 0.0
    properties: Dict = field(default_factory=dict)
    evidence: str = ""
    section: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ExtractionResult:
    """Result of entity extraction for a document."""
    document_id: str
    entities: List[ExtractedEntity]
    relationships: List[ExtractedRelationship]
    extraction_time: float
    method_used: str
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "document_id": self.document_id,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "extraction_time": self.extraction_time,
            "method_used": self.method_used,
            "metadata": self.metadata,
            "summary": {
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "entities_by_type": self._count_by_type(self.entities),
                "relationships_by_type": self._count_by_type(self.relationships, "relationship_type")
            }
        }
    
    def _count_by_type(self, items: List, type_attr: str = "entity_type") -> Dict[str, int]:
        counts = {}
        for item in items:
            t = getattr(item, type_attr)
            counts[t] = counts.get(t, 0) + 1
        return counts


class EntityExtractor:
    """
    Main entity extraction class.
    Uses Groq LLM for primary extraction with spaCy fallback.
    """
    
    # System prompt for entity extraction
    EXTRACTION_PROMPT = """You are a financial document analyst specializing in SEC 10-K filings fraud detection.
Extract entities and relationships from the following text.

For ENTITIES, extract:
1. COMPANY: Parent companies, subsidiaries, joint ventures, related parties
2. PERSON: Executives, board members, auditors, key employees
3. FINANCIAL: Revenue sources, expenses, assets, liabilities, monetary values
4. TEMPORAL: Dates, fiscal periods, timelines
5. REGULATORY: SEC references, compliance items, audit opinions

For RELATIONSHIPS, focus on:
1. Ownership/control relationships (OWNS, CONTROLS, SUBSIDIARY_OF)
2. Transaction relationships (HAS_TRANSACTION_WITH, PAYS, LOANS_TO)
3. Employment relationships (EMPLOYS, BOARD_MEMBER_OF)
4. Disclosure relationships (DISCLOSED_IN, NOT_DISCLOSED_IN)

Pay special attention to potential fraud indicators:
- Undisclosed related party transactions
- Hidden subsidiaries or shell companies
- Unusual executive compensation
- Auditor concerns or changes

Return a JSON object with this exact structure:
{
    "entities": [
        {
            "text": "entity text",
            "entity_type": "COMPANY|PERSON|FINANCIAL|TEMPORAL|REGULATORY",
            "sub_type": "more specific type",
            "confidence": 0.0-1.0,
            "attributes": {},
            "fraud_indicators": []
        }
    ],
    "relationships": [
        {
            "source": "source entity text",
            "target": "target entity text",
            "relationship_type": "OWNS|CONTROLS|etc",
            "confidence": 0.0-1.0,
            "properties": {}
        }
    ]
}

TEXT TO ANALYZE:
"""
    
    def __init__(
        self,
        use_llm: bool = True,
        use_spacy_fallback: bool = True,
        config_path: Optional[str] = None
    ):
        """
        Initialize entity extractor.
        
        Args:
            use_llm: Whether to use LLM for extraction
            use_spacy_fallback: Whether to use spaCy as fallback
            config_path: Path to configuration file
        """
        self.logger = get_nlp_logger()
        self.use_llm = use_llm
        self.use_spacy_fallback = use_spacy_fallback
        
        # Load configuration
        try:
            self.config = get_config(config_path)
            self.llm_config = self.config.llm_config
        except Exception:
            self.config = None
            self.llm_config = {}
        
        # Initialize components
        self._groq_client = None
        self._spacy_nlp = None
        self._entity_counter = 0
        self._relationship_counter = 0
        
        # Load entity type definitions
        self._load_entity_definitions()
        
        self.logger.info("EntityExtractor initialized")
    
    def _load_entity_definitions(self) -> None:
        """Load entity type definitions from YAML."""
        try:
            config_dir = Path(__file__).parent.parent.parent / "config"
            entity_file = config_dir / "entity_types.yaml"
            
            if entity_file.exists():
                with open(entity_file, 'r', encoding='utf-8') as f:
                    self.entity_definitions = yaml.safe_load(f)
            else:
                self.entity_definitions = {}
                
        except Exception as e:
            self.logger.warning(f"Could not load entity definitions: {e}")
            self.entity_definitions = {}
    
    def _init_groq_client(self) -> None:
        """Initialize Groq client."""
        if self._groq_client is not None:
            return
        
        try:
            from groq import Groq
            import os
            
            api_key = os.getenv("GROQ_API_KEY", self.llm_config.get("api_key", ""))
            
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment")
            
            self._groq_client = Groq(api_key=api_key)
            self.logger.info("Groq client initialized")
            
        except ImportError:
            self.logger.error("groq package not installed")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize Groq: {e}")
            raise
    
    def _init_spacy(self) -> None:
        """Initialize spaCy NLP."""
        if self._spacy_nlp is not None:
            return
        
        try:
            import spacy
            
            model_name = self.config.get("entity_extraction.spacy_model", "en_core_web_lg") if self.config else "en_core_web_lg"
            
            try:
                self._spacy_nlp = spacy.load(model_name)
            except OSError:
                self.logger.warning(f"Downloading spaCy model: {model_name}")
                spacy.cli.download(model_name)
                self._spacy_nlp = spacy.load(model_name)
            
            self.logger.info(f"spaCy model loaded: {model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize spaCy: {e}")
            raise
    
    def _generate_entity_id(self) -> str:
        """Generate unique entity ID."""
        self._entity_counter += 1
        return f"ENT_{self._entity_counter:06d}"
    
    def _generate_relationship_id(self) -> str:
        """Generate unique relationship ID."""
        self._relationship_counter += 1
        return f"REL_{self._relationship_counter:06d}"
    
    def extract_with_llm(
        self,
        text: str,
        section: str = "",
        page_number: Optional[int] = None
    ) -> Tuple[List[ExtractedEntity], List[ExtractedRelationship]]:
        """
        Extract entities using Groq LLM with rate limiting.
        
        Args:
            text: Text to extract from
            section: Section name
            page_number: Page number
        
        Returns:
            Tuple of (entities, relationships)
        """
        import time
        
        self._init_groq_client()
        
        entities = []
        relationships = []
        
        # Get rate limit settings
        rate_limit_config = self.llm_config.get("rate_limit", {})
        delay_between_requests = rate_limit_config.get("delay_between_requests", 2.5)
        max_retries = self.llm_config.get("retry_attempts", 3)
        
        # Add delay before API call to respect rate limits
        if hasattr(self, '_last_api_call_time'):
            time_since_last_call = time.time() - self._last_api_call_time
            if time_since_last_call < delay_between_requests:
                sleep_time = delay_between_requests - time_since_last_call
                self.logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        
        for attempt in range(max_retries):
            try:
                # Prepare prompt
                prompt = self.EXTRACTION_PROMPT + text[:8000]  # Limit text length
                
                # Call Groq API
                model = self.llm_config.get("model", "llama-3.1-8b-instant")
                
                response = self._groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a financial document analysis expert. Always respond with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=self.llm_config.get("max_tokens", 2048)
                )
                
                # Track API call time
                self._last_api_call_time = time.time()
                
                # Parse response
                response_text = response.choices[0].message.content
                
                # Extract JSON from response
                json_match = re.search(r'\{[\s\S]*\}', response_text)
                if json_match:
                    result = json.loads(json_match.group())
                    
                    # Process entities
                    for ent_data in result.get("entities", []):
                        entity = ExtractedEntity(
                            id=self._generate_entity_id(),
                            text=ent_data.get("text", ""),
                            entity_type=ent_data.get("entity_type", "UNKNOWN"),
                            sub_type=ent_data.get("sub_type"),
                            confidence=float(ent_data.get("confidence", 0.8)),
                            section=section,
                            page_number=page_number,
                            attributes=ent_data.get("attributes", {}),
                            fraud_indicators=ent_data.get("fraud_indicators", [])
                        )
                        entities.append(entity)
                    
                    # Process relationships
                    # Build entity lookup by text
                    entity_lookup = {e.text.lower(): e.id for e in entities}
                    
                    for rel_data in result.get("relationships", []):
                        source_text = rel_data.get("source", "").lower()
                        target_text = rel_data.get("target", "").lower()
                        
                        source_id = entity_lookup.get(source_text)
                        target_id = entity_lookup.get(target_text)
                        
                        if source_id and target_id:
                            relationship = ExtractedRelationship(
                                id=self._generate_relationship_id(),
                                source_entity_id=source_id,
                                target_entity_id=target_id,
                                relationship_type=rel_data.get("relationship_type", "RELATED_TO"),
                                confidence=float(rel_data.get("confidence", 0.7)),
                                properties=rel_data.get("properties", {}),
                                section=section
                            )
                            relationships.append(relationship)
                
                self.logger.debug(f"LLM extracted {len(entities)} entities, {len(relationships)} relationships")
                break  # Success, exit retry loop
                
            except Exception as e:
                error_message = str(e)
                
                # Check if it's a rate limit error
                if "rate_limit" in error_message.lower() or "429" in error_message:
                    # Extract wait time from error message if available
                    wait_time_match = re.search(r'try again in (\d+(?:\.\d+)?)', error_message)
                    if wait_time_match:
                        wait_time = float(wait_time_match.group(1)) + 1  # Add 1 second buffer
                    else:
                        # Exponential backoff
                        wait_time = (2 ** attempt) * 2  # 2s, 4s, 8s
                    
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Rate limit hit, retrying in {wait_time:.1f}s... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        self.logger.error(f"Rate limit exceeded after {max_retries} attempts, falling back to spaCy")
                        raise
                else:
                    self.logger.error(f"LLM extraction failed: {e}")
                    raise
        
        return entities, relationships
    
    def extract_with_spacy(
        self,
        text: str,
        section: str = "",
        page_number: Optional[int] = None
    ) -> List[ExtractedEntity]:
        """
        Extract entities using spaCy NER.
        
        Args:
            text: Text to extract from
            section: Section name
            page_number: Page number
        
        Returns:
            List of extracted entities
        """
        self._init_spacy()
        
        entities = []
        doc = self._spacy_nlp(text)
        
        # Map spaCy labels to our types
        label_mapping = {
            "ORG": "COMPANY",
            "PERSON": "PERSON",
            "MONEY": "FINANCIAL",
            "PERCENT": "FINANCIAL",
            "DATE": "TEMPORAL",
            "TIME": "TEMPORAL",
            "GPE": "LOCATION",
            "LAW": "REGULATORY"
        }
        
        for ent in doc.ents:
            entity_type = label_mapping.get(ent.label_, "OTHER")
            
            entity = ExtractedEntity(
                id=self._generate_entity_id(),
                text=ent.text,
                entity_type=entity_type,
                sub_type=ent.label_,
                confidence=0.7,
                start_position=ent.start_char,
                end_position=ent.end_char,
                section=section,
                page_number=page_number
            )
            entities.append(entity)
        
        self.logger.debug(f"spaCy extracted {len(entities)} entities")
        return entities
    
    def extract_patterns(
        self,
        text: str,
        section: str = "",
        page_number: Optional[int] = None
    ) -> List[ExtractedEntity]:
        """
        Extract entities using regex patterns.
        
        Args:
            text: Text to extract from
            section: Section name
            page_number: Page number
        
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Pattern definitions
        patterns = {
            "MONETARY_VALUE": r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|M|B|K))?",
            "PERCENTAGE": r"\d+(?:\.\d+)?%",
            "FISCAL_YEAR": r"(?:fiscal\s+)?(?:year\s+)?(?:FY\s*)?\d{4}",
            "FISCAL_QUARTER": r"Q[1-4]\s+\d{4}|(?:first|second|third|fourth)\s+quarter\s+\d{4}",
            "DATE": r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
            "SEC_ITEM": r"ITEM\s+\d+[A-Z]?\.?\s*[A-Z\s]+",
        }
        
        for entity_type, pattern in patterns.items():
            for match in re.finditer(pattern, text, re.IGNORECASE):
                entity = ExtractedEntity(
                    id=self._generate_entity_id(),
                    text=match.group(),
                    entity_type="FINANCIAL" if entity_type in ["MONETARY_VALUE", "PERCENTAGE"] else "TEMPORAL",
                    sub_type=entity_type,
                    confidence=0.9,
                    start_position=match.start(),
                    end_position=match.end(),
                    section=section,
                    page_number=page_number
                )
                entities.append(entity)
        
        return entities
    
    def merge_entities(
        self,
        entity_lists: List[List[ExtractedEntity]],
        similarity_threshold: float = 0.85
    ) -> List[ExtractedEntity]:
        """
        Merge entities from multiple extraction methods.
        
        Args:
            entity_lists: List of entity lists to merge
            similarity_threshold: Threshold for considering entities similar
        
        Returns:
            Merged list of entities
        """
        all_entities = []
        for entity_list in entity_lists:
            all_entities.extend(entity_list)
        
        # Simple deduplication by text
        seen_texts = {}
        merged = []
        
        for entity in all_entities:
            text_key = entity.text.lower().strip()
            
            if text_key not in seen_texts:
                seen_texts[text_key] = entity
                merged.append(entity)
            else:
                # Keep entity with higher confidence
                existing = seen_texts[text_key]
                if entity.confidence > existing.confidence:
                    merged.remove(existing)
                    merged.append(entity)
                    seen_texts[text_key] = entity
        
        return merged
    
    def extract(
        self,
        text: str,
        document_id: str = "",
        section: str = "",
        page_number: Optional[int] = None
    ) -> ExtractionResult:
        """
        Main extraction method - combines all extraction strategies.
        
        Args:
            text: Text to extract from
            document_id: Document identifier
            section: Section name
            page_number: Page number
        
        Returns:
            ExtractionResult object
        """
        start_time = datetime.now()
        all_entities = []
        all_relationships = []
        method_used = []
        
        # Try LLM extraction
        if self.use_llm:
            try:
                llm_entities, llm_relationships = self.extract_with_llm(
                    text, section, page_number
                )
                all_entities.append(llm_entities)
                all_relationships.extend(llm_relationships)
                method_used.append("llm")
            except Exception as e:
                self.logger.warning(f"LLM extraction failed, trying fallback: {e}")
        
        # spaCy fallback or supplementary
        if self.use_spacy_fallback:
            try:
                spacy_entities = self.extract_with_spacy(text, section, page_number)
                all_entities.append(spacy_entities)
                method_used.append("spacy")
            except Exception as e:
                self.logger.warning(f"spaCy extraction failed: {e}")
        
        # Pattern-based extraction (always run)
        pattern_entities = self.extract_patterns(text, section, page_number)
        all_entities.append(pattern_entities)
        method_used.append("patterns")
        
        # Merge entities
        merged_entities = self.merge_entities(all_entities)
        
        extraction_time = (datetime.now() - start_time).total_seconds()
        
        return ExtractionResult(
            document_id=document_id,
            entities=merged_entities,
            relationships=all_relationships,
            extraction_time=extraction_time,
            method_used="+".join(method_used),
            metadata={
                "text_length": len(text),
                "section": section,
                "page_number": page_number
            }
        )
    
    def extract_from_chunks(
        self,
        chunks: List[Dict],
        document_id: str = ""
    ) -> ExtractionResult:
        """
        Extract entities from multiple text chunks.
        
        Args:
            chunks: List of chunk dictionaries with 'text', 'section_type', 'page_number'
            document_id: Document identifier
        
        Returns:
            Combined ExtractionResult
        """
        start_time = datetime.now()
        all_entities = []
        all_relationships = []
        
        for chunk in chunks:
            text = chunk.get("text", "")
            section = chunk.get("section_type", "")
            page = chunk.get("page_number")
            
            result = self.extract(text, document_id, section, page)
            all_entities.extend(result.entities)
            all_relationships.extend(result.relationships)
        
        # Merge duplicate entities across chunks
        merged_entities = self.merge_entities([all_entities])
        
        extraction_time = (datetime.now() - start_time).total_seconds()
        
        return ExtractionResult(
            document_id=document_id,
            entities=merged_entities,
            relationships=all_relationships,
            extraction_time=extraction_time,
            method_used="multi_chunk",
            metadata={
                "chunk_count": len(chunks)
            }
        )
