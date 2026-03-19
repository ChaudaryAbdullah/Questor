"""
Corporate Entity Extractor
Specializes in extracting corporate structure entities like subsidiaries, 
parent companies, related parties, and shell companies.
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ..utils.logger import get_nlp_logger


@dataclass
class CorporateEntity:
    """Represents a corporate entity."""
    name: str
    entity_type: str  # PARENT_COMPANY, SUBSIDIARY, JOINT_VENTURE, RELATED_PARTY, SHELL_COMPANY
    ownership_percentage: Optional[float] = None
    jurisdiction: Optional[str] = None
    consolidation_status: Optional[str] = None
    mentioned_sections: List[str] = None
    fraud_indicators: List[str] = None
    confidence: float = 0.0
    context: str = ""
    
    def __post_init__(self):
        if self.mentioned_sections is None:
            self.mentioned_sections = []
        if self.fraud_indicators is None:
            self.fraud_indicators = []


class CorporateExtractor:
    """
    Extractor for corporate structure entities.
    Identifies parent companies, subsidiaries, joint ventures, and related parties.
    """
    
    # Patterns for corporate entities
    SUBSIDIARY_PATTERNS = [
        r"(?:our\s+)?(?:wholly[- ]owned\s+)?subsidiary(?:,?\s*)([A-Z][A-Za-z\s\.,&]+(?:Inc|Corp|LLC|Ltd|Co|Company|LP)?\.?)",
        r"([A-Z][A-Za-z\s\.,&]+(?:Inc|Corp|LLC|Ltd|Co)?\.?)\s*(?:,\s*)?(?:a|our)\s+(?:wholly[- ]owned\s+)?subsidiary",
        r"subsidiary\s+(?:of\s+)?(?:the\s+)?(?:Company\s+)?(?:named\s+)?([A-Z][A-Za-z\s\.,&]+)",
    ]
    
    RELATED_PARTY_PATTERNS = [
        r"related\s+part(?:y|ies)\s+(?:transaction|relationship)?\s*(?:with\s+)?([A-Z][A-Za-z\s\.,&]+)",
        r"([A-Z][A-Za-z\s\.,&]+)\s*(?:,\s*)?(?:a\s+)?related\s+party",
        r"affiliate(?:d)?\s+(?:entity|company|with)\s*([A-Z][A-Za-z\s\.,&]+)",
    ]
    
    SHELL_COMPANY_PATTERNS = [
        r"special\s+purpose\s+(?:entity|vehicle|company)\s*(?:named\s+)?([A-Z][A-Za-z\s\.,&]+)?",
        r"(?:SPE|SPV)\s+(?:named\s+)?([A-Z][A-Za-z\s\.,&]+)?",
        r"variable\s+interest\s+entity\s*(?:named\s+)?([A-Z][A-Za-z\s\.,&]+)?",
        r"VIE\s+([A-Z][A-Za-z\s\.,&]+)?",
        r"off[- ]balance\s+sheet\s+(?:entity|arrangement|vehicle)",
    ]
    
    JOINT_VENTURE_PATTERNS = [
        r"joint\s+venture\s+(?:with\s+)?(?:named\s+)?([A-Z][A-Za-z\s\.,&]+)?",
        r"([A-Z][A-Za-z\s\.,&]+)\s+joint\s+venture",
        r"partnership\s+with\s+([A-Z][A-Za-z\s\.,&]+)",
        r"strategic\s+alliance\s+with\s+([A-Z][A-Za-z\s\.,&]+)",
    ]
    
    PARENT_COMPANY_PATTERNS = [
        r"(?:our\s+)?parent\s+company\s*(?:,?\s*)([A-Z][A-Za-z\s\.,&]+)?",
        r"([A-Z][A-Za-z\s\.,&]+)\s+(?:is\s+)?our\s+parent",
        r"controlled\s+by\s+([A-Z][A-Za-z\s\.,&]+)",
    ]
    
    OWNERSHIP_PATTERN = r"(\d+(?:\.\d+)?)\s*(?:%|percent)"
    
    # Jurisdiction indicators
    JURISDICTIONS = [
        "Delaware", "Nevada", "California", "New York", "Texas",
        "Cayman Islands", "British Virgin Islands", "Bermuda",
        "Luxembourg", "Netherlands", "Ireland", "Singapore", "Hong Kong"
    ]
    
    def __init__(self):
        self.logger = get_nlp_logger()
        
        # Compile patterns
        self.compiled_patterns = {
            "SUBSIDIARY": [re.compile(p, re.IGNORECASE) for p in self.SUBSIDIARY_PATTERNS],
            "RELATED_PARTY": [re.compile(p, re.IGNORECASE) for p in self.RELATED_PARTY_PATTERNS],
            "SHELL_COMPANY": [re.compile(p, re.IGNORECASE) for p in self.SHELL_COMPANY_PATTERNS],
            "JOINT_VENTURE": [re.compile(p, re.IGNORECASE) for p in self.JOINT_VENTURE_PATTERNS],
            "PARENT_COMPANY": [re.compile(p, re.IGNORECASE) for p in self.PARENT_COMPANY_PATTERNS],
        }
        
        self.ownership_regex = re.compile(self.OWNERSHIP_PATTERN)
    
    def extract(
        self,
        text: str,
        section: str = ""
    ) -> List[CorporateEntity]:
        """
        Extract corporate entities from text.
        
        Args:
            text: Text to analyze
            section: Section name for context
        
        Returns:
            List of CorporateEntity objects
        """
        entities = []
        
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    # Get entity name from capture group
                    name = None
                    for group in match.groups():
                        if group and len(group.strip()) > 2:
                            name = self._clean_entity_name(group)
                            break
                    
                    if not name:
                        continue
                    
                    # Get context
                    start = max(0, match.start() - 100)
                    end = min(len(text), match.end() + 100)
                    context = text[start:end]
                    
                    # Extract ownership percentage if present
                    ownership = self._extract_ownership(context)
                    
                    # Extract jurisdiction
                    jurisdiction = self._extract_jurisdiction(context)
                    
                    # Check for fraud indicators
                    fraud_indicators = self._check_fraud_indicators(
                        name, entity_type, context, section
                    )
                    
                    entity = CorporateEntity(
                        name=name,
                        entity_type=entity_type,
                        ownership_percentage=ownership,
                        jurisdiction=jurisdiction,
                        mentioned_sections=[section] if section else [],
                        fraud_indicators=fraud_indicators,
                        confidence=0.8 if ownership else 0.7,
                        context=context.strip()
                    )
                    entities.append(entity)
        
        # Deduplicate by name
        unique_entities = self._deduplicate(entities)
        
        self.logger.debug(f"Extracted {len(unique_entities)} corporate entities")
        return unique_entities
    
    def _clean_entity_name(self, name: str) -> str:
        """Clean up extracted entity name."""
        # Remove common noise
        name = name.strip()
        name = re.sub(r'^(?:the|a|an)\s+', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name)
        
        # Remove trailing punctuation except periods after Inc., etc.
        name = re.sub(r'[,\s]+$', '', name)
        
        return name
    
    def _extract_ownership(self, context: str) -> Optional[float]:
        """Extract ownership percentage from context."""
        match = self.ownership_regex.search(context)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None
    
    def _extract_jurisdiction(self, context: str) -> Optional[str]:
        """Extract jurisdiction from context."""
        for jurisdiction in self.JURISDICTIONS:
            if jurisdiction.lower() in context.lower():
                return jurisdiction
        return None
    
    def _check_fraud_indicators(
        self,
        name: str,
        entity_type: str,
        context: str,
        section: str
    ) -> List[str]:
        """Check for potential fraud indicators."""
        indicators = []
        context_lower = context.lower()
        
        # Shell company red flags
        if entity_type == "SHELL_COMPANY":
            indicators.append("SHELL_COMPANY_STRUCTURE")
            
            if any(tax_haven in context_lower for tax_haven in 
                   ["cayman", "virgin islands", "bermuda", "luxembourg"]):
                indicators.append("TAX_HAVEN_JURISDICTION")
        
        # Related party red flags
        if entity_type == "RELATED_PARTY":
            if "undisclosed" in context_lower or "not previously" in context_lower:
                indicators.append("POTENTIALLY_UNDISCLOSED")
            
            if "material" in context_lower:
                indicators.append("MATERIAL_RELATED_PARTY")
        
        # Subsidiary red flags
        if entity_type == "SUBSIDIARY":
            if "unconsolidated" in context_lower:
                indicators.append("UNCONSOLIDATED_SUBSIDIARY")
            
            if "off-balance" in context_lower or "off balance" in context_lower:
                indicators.append("OFF_BALANCE_SHEET")
        
        # Section-based indicators
        if section and "footnote" not in section.lower():
            if entity_type in ["SUBSIDIARY", "RELATED_PARTY"]:
                indicators.append("CHECK_FOOTNOTE_DISCLOSURE")
        
        return indicators
    
    def _deduplicate(
        self,
        entities: List[CorporateEntity]
    ) -> List[CorporateEntity]:
        """Deduplicate entities by name, keeping most informative."""
        seen = {}
        
        for entity in entities:
            key = entity.name.lower()
            
            if key not in seen:
                seen[key] = entity
            else:
                # Merge information
                existing = seen[key]
                
                # Keep ownership if found
                if entity.ownership_percentage and not existing.ownership_percentage:
                    existing.ownership_percentage = entity.ownership_percentage
                
                # Keep jurisdiction if found
                if entity.jurisdiction and not existing.jurisdiction:
                    existing.jurisdiction = entity.jurisdiction
                
                # Merge sections
                for section in entity.mentioned_sections:
                    if section not in existing.mentioned_sections:
                        existing.mentioned_sections.append(section)
                
                # Merge fraud indicators
                for indicator in entity.fraud_indicators:
                    if indicator not in existing.fraud_indicators:
                        existing.fraud_indicators.append(indicator)
                
                # Update confidence
                existing.confidence = max(existing.confidence, entity.confidence)
        
        return list(seen.values())
    
    def find_relationships(
        self,
        entities: List[CorporateEntity],
        text: str
    ) -> List[Dict]:
        """
        Find relationships between corporate entities.
        
        Args:
            entities: List of extracted entities
            text: Original text for context
        
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Look for ownership relationships
        ownership_patterns = [
            r"([A-Z][A-Za-z\s\.,&]+)\s+owns\s+(\d+(?:\.\d+)?%?)\s+(?:of\s+)?([A-Z][A-Za-z\s\.,&]+)",
            r"([A-Z][A-Za-z\s\.,&]+)\s+(?:is\s+)?(?:a\s+)?subsidiary\s+of\s+([A-Z][A-Za-z\s\.,&]+)",
            r"([A-Z][A-Za-z\s\.,&]+)\s+controls\s+([A-Z][A-Za-z\s\.,&]+)",
        ]
        
        entity_names = {e.name.lower() for e in entities}
        
        for pattern in ownership_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    source = self._clean_entity_name(groups[0])
                    target = self._clean_entity_name(groups[-1])
                    
                    if source.lower() in entity_names or target.lower() in entity_names:
                        relationships.append({
                            "source": source,
                            "target": target,
                            "relationship_type": "OWNS" if "owns" in match.group().lower() else "CONTROLS",
                            "evidence": match.group()
                        })
        
        return relationships
    
    def analyze_disclosure_gaps(
        self,
        entities: List[CorporateEntity],
        sections_found: List[str]
    ) -> List[Dict]:
        """
        Analyze potential disclosure gaps for entities.
        
        Args:
            entities: List of extracted entities
            sections_found: List of sections where entities were mentioned
        
        Returns:
            List of disclosure gap findings
        """
        gaps = []
        
        for entity in entities:
            # Subsidiaries should be in footnotes
            if entity.entity_type == "SUBSIDIARY":
                if not any("footnote" in s.lower() or "note" in s.lower() 
                          for s in entity.mentioned_sections):
                    gaps.append({
                        "entity": entity.name,
                        "entity_type": entity.entity_type,
                        "issue": "SUBSIDIARY_NOT_IN_FOOTNOTES",
                        "severity": "HIGH",
                        "detail": f"Subsidiary '{entity.name}' mentioned in text but may not be disclosed in footnotes"
                    })
            
            # Related parties need proper disclosure
            if entity.entity_type == "RELATED_PARTY":
                gaps.append({
                    "entity": entity.name,
                    "entity_type": entity.entity_type,
                    "issue": "VERIFY_RELATED_PARTY_DISCLOSURE",
                    "severity": "MEDIUM",
                    "detail": f"Related party '{entity.name}' requires verification of complete disclosure"
                })
            
            # Shell companies are high risk
            if entity.entity_type == "SHELL_COMPANY":
                gaps.append({
                    "entity": entity.name,
                    "entity_type": entity.entity_type,
                    "issue": "SHELL_COMPANY_REVIEW",
                    "severity": "CRITICAL",
                    "detail": f"Special purpose entity '{entity.name}' requires detailed consolidation review"
                })
        
        return gaps
