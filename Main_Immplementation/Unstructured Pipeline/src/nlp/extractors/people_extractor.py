"""
People Entity Extractor
Specializes in extracting executives, board members, auditors, and family relationships.
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from ..utils.logger import get_nlp_logger


@dataclass
class PersonEntity:
    """Represents a person entity."""
    name: str
    entity_type: str  # EXECUTIVE, BOARD_MEMBER, KEY_EMPLOYEE, AUDITOR, FAMILY_RELATION
    title: Optional[str] = None
    roles: List[str] = field(default_factory=list)
    company: Optional[str] = None
    related_to: Optional[str] = None
    relationship_type: Optional[str] = None
    mentioned_sections: List[str] = field(default_factory=list)
    fraud_indicators: List[str] = field(default_factory=list)
    confidence: float = 0.0
    context: str = ""


class PeopleExtractor:
    """
    Extractor for people entities.
    Identifies executives, board members, auditors, and family relationships.
    """
    
    # Executive title patterns
    EXECUTIVE_TITLES = [
        r"Chief\s+Executive\s+Officer",
        r"Chief\s+Financial\s+Officer", 
        r"Chief\s+Operating\s+Officer",
        r"Chief\s+Technology\s+Officer",
        r"Chief\s+Information\s+Officer",
        r"Chief\s+Accounting\s+Officer",
        r"Chief\s+Legal\s+Officer",
        r"Chief\s+Risk\s+Officer",
        r"CEO", r"CFO", r"COO", r"CTO", r"CIO", r"CAO", r"CLO", r"CRO",
        r"President(?:\s+and\s+CEO)?",
        r"Vice\s+President",
        r"Executive\s+Vice\s+President",
        r"Senior\s+Vice\s+President",
        r"Controller",
        r"Treasurer",
        r"General\s+Counsel",
        r"Secretary",
    ]
    
    BOARD_TITLES = [
        r"Chairman(?:\s+of\s+the\s+Board)?",
        r"Vice\s+Chairman",
        r"Lead\s+Independent\s+Director",
        r"Independent\s+Director",
        r"Director",
        r"Board\s+Member",
        r"Audit\s+Committee\s+(?:Chair|Member)",
        r"Compensation\s+Committee\s+(?:Chair|Member)",
        r"Nominating\s+Committee\s+(?:Chair|Member)",
        r"Governance\s+Committee\s+(?:Chair|Member)",
    ]
    
    AUDITOR_PATTERNS = [
        r"(Deloitte(?:\s+&\s+Touche)?(?:\s+LLP)?)",
        r"(Ernst\s*&\s*Young(?:\s+LLP)?|EY(?:\s+LLP)?)",
        r"(KPMG(?:\s+LLP)?)",
        r"(PricewaterhouseCoopers(?:\s+LLP)?|PwC(?:\s+LLP)?)",
        r"(BDO(?:\s+USA)?(?:\s+LLP)?)",
        r"(Grant\s+Thornton(?:\s+LLP)?)",
        r"(RSM(?:\s+US)?(?:\s+LLP)?)",
        r"(Crowe(?:\s+LLP)?)",
        r"independent\s+registered\s+public\s+accounting\s+firm",
        r"independent\s+auditor",
    ]
    
    FAMILY_PATTERNS = [
        r"(?:his|her|their)\s+(?:spouse|wife|husband)",
        r"(?:son|daughter|child|children)\s+of",
        r"(?:brother|sister|sibling)\s+of",
        r"(?:father|mother|parent)\s+of",
        r"immediate\s+family\s+member",
        r"family\s+member",
        r"related\s+(?:by\s+marriage|through)",
    ]
    
    # Name pattern - captures typical Western names
    NAME_PATTERN = r"([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:\s+(?:Jr\.|Sr\.|III|IV|II))?)"
    
    def __init__(self):
        self.logger = get_nlp_logger()
        
        # Compile combined title patterns
        self.exec_pattern = re.compile(
            f"({self.NAME_PATTERN})\\s*,?\\s*(?:our\\s+)?({'|'.join(self.EXECUTIVE_TITLES)})",
            re.IGNORECASE
        )
        
        self.exec_pattern_reverse = re.compile(
            f"({'|'.join(self.EXECUTIVE_TITLES)})\\s*,?\\s*({self.NAME_PATTERN})",
            re.IGNORECASE
        )
        
        self.board_pattern = re.compile(
            f"({self.NAME_PATTERN})\\s*,?\\s*(?:our\\s+)?(?:a\\s+)?({'|'.join(self.BOARD_TITLES)})",
            re.IGNORECASE
        )
        
        self.auditor_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.AUDITOR_PATTERNS
        ]
        
        self.family_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.FAMILY_PATTERNS
        ]
    
    def extract(
        self,
        text: str,
        section: str = ""
    ) -> List[PersonEntity]:
        """
        Extract people entities from text.
        
        Args:
            text: Text to analyze
            section: Section name for context
        
        Returns:
            List of PersonEntity objects
        """
        entities = []
        
        # Extract executives
        entities.extend(self._extract_executives(text, section))
        
        # Extract board members
        entities.extend(self._extract_board_members(text, section))
        
        # Extract auditors
        entities.extend(self._extract_auditors(text, section))
        
        # Extract family relationships
        entities.extend(self._extract_family_relationships(text, section))
        
        # Deduplicate
        unique_entities = self._deduplicate(entities)
        
        self.logger.debug(f"Extracted {len(unique_entities)} people entities")
        return unique_entities
    
    def _extract_executives(
        self,
        text: str,
        section: str
    ) -> List[PersonEntity]:
        """Extract executive entities."""
        entities = []
        
        # Pattern: Name, Title
        for match in self.exec_pattern.finditer(text):
            name = self._clean_name(match.group(1))
            title = match.group(2).strip()
            
            context = self._get_context(text, match.start(), match.end())
            fraud_indicators = self._check_executive_fraud_indicators(context)
            
            entities.append(PersonEntity(
                name=name,
                entity_type="EXECUTIVE",
                title=title,
                roles=[title],
                mentioned_sections=[section] if section else [],
                fraud_indicators=fraud_indicators,
                confidence=0.85,
                context=context
            ))
        
        # Pattern: Title Name
        for match in self.exec_pattern_reverse.finditer(text):
            title = match.group(1).strip()
            name = self._clean_name(match.group(2))
            
            context = self._get_context(text, match.start(), match.end())
            fraud_indicators = self._check_executive_fraud_indicators(context)
            
            entities.append(PersonEntity(
                name=name,
                entity_type="EXECUTIVE",
                title=title,
                roles=[title],
                mentioned_sections=[section] if section else [],
                fraud_indicators=fraud_indicators,
                confidence=0.85,
                context=context
            ))
        
        return entities
    
    def _extract_board_members(
        self,
        text: str,
        section: str
    ) -> List[PersonEntity]:
        """Extract board member entities."""
        entities = []
        
        for match in self.board_pattern.finditer(text):
            name = self._clean_name(match.group(1))
            role = match.group(2).strip()
            
            context = self._get_context(text, match.start(), match.end())
            
            # Check for independence
            is_independent = "independent" in context.lower()
            
            fraud_indicators = []
            if not is_independent and "audit committee" in role.lower():
                fraud_indicators.append("NON_INDEPENDENT_AUDIT_COMMITTEE")
            
            entities.append(PersonEntity(
                name=name,
                entity_type="BOARD_MEMBER",
                title=role,
                roles=[role],
                mentioned_sections=[section] if section else [],
                fraud_indicators=fraud_indicators,
                confidence=0.8,
                context=context
            ))
        
        return entities
    
    def _extract_auditors(
        self,
        text: str,
        section: str
    ) -> List[PersonEntity]:
        """Extract auditor entities."""
        entities = []
        
        for pattern in self.auditor_patterns:
            for match in pattern.finditer(text):
                name = match.group(0) if not match.groups() else match.group(1)
                name = self._clean_name(name)
                
                context = self._get_context(text, match.start(), match.end())
                fraud_indicators = self._check_auditor_fraud_indicators(context)
                
                entities.append(PersonEntity(
                    name=name,
                    entity_type="AUDITOR",
                    title="Independent Auditor",
                    roles=["External Auditor"],
                    mentioned_sections=[section] if section else [],
                    fraud_indicators=fraud_indicators,
                    confidence=0.9,
                    context=context
                ))
        
        return entities
    
    def _extract_family_relationships(
        self,
        text: str,
        section: str
    ) -> List[PersonEntity]:
        """Extract family relationship entities."""
        entities = []
        
        for pattern in self.family_patterns:
            for match in pattern.finditer(text):
                # Look for a name near the family mention
                start = max(0, match.start() - 100)
                end = min(len(text), match.end() + 100)
                context = text[start:end]
                
                # Try to find associated names
                name_matches = re.findall(self.NAME_PATTERN, context)
                
                relationship_type = match.group(0).lower()
                
                for name in name_matches[:2]:  # Limit to avoid noise
                    entities.append(PersonEntity(
                        name=self._clean_name(name),
                        entity_type="FAMILY_RELATION",
                        relationship_type=relationship_type,
                        mentioned_sections=[section] if section else [],
                        fraud_indicators=["POTENTIAL_RELATED_PARTY"],
                        confidence=0.6,
                        context=context
                    ))
        
        return entities
    
    def _clean_name(self, name: str) -> str:
        """Clean up extracted name."""
        name = name.strip()
        name = re.sub(r'\s+', ' ', name)
        name = re.sub(r'[,\s]+$', '', name)
        return name
    
    def _get_context(
        self,
        text: str,
        start: int,
        end: int,
        window: int = 150
    ) -> str:
        """Get context around match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end].strip()
    
    def _check_executive_fraud_indicators(self, context: str) -> List[str]:
        """Check for executive-related fraud indicators."""
        indicators = []
        context_lower = context.lower()
        
        # Related party transactions
        if "related party" in context_lower or "related transaction" in context_lower:
            indicators.append("EXECUTIVE_RELATED_PARTY_TRANSACTION")
        
        # Compensation concerns
        if any(word in context_lower for word in ["bonus", "incentive", "stock option"]):
            if any(word in context_lower for word in ["material", "significant", "substantial"]):
                indicators.append("MATERIAL_EXECUTIVE_COMPENSATION")
        
        # Control concerns
        if "significant control" in context_lower or "voting control" in context_lower:
            indicators.append("EXECUTIVE_CONTROL_CONCENTRATION")
        
        return indicators
    
    def _check_auditor_fraud_indicators(self, context: str) -> List[str]:
        """Check for auditor-related fraud indicators."""
        indicators = []
        context_lower = context.lower()
        
        # Qualified opinion
        if "qualified" in context_lower and "opinion" in context_lower:
            indicators.append("QUALIFIED_AUDIT_OPINION")
        
        # Going concern
        if "going concern" in context_lower:
            indicators.append("GOING_CONCERN_DOUBT")
        
        # Material weakness
        if "material weakness" in context_lower:
            indicators.append("MATERIAL_WEAKNESS_IN_CONTROLS")
        
        # Auditor change
        if any(word in context_lower for word in ["former auditor", "replaced", "dismissed", "resigned"]):
            indicators.append("AUDITOR_CHANGE")
        
        # Adverse opinion
        if "adverse" in context_lower and "opinion" in context_lower:
            indicators.append("ADVERSE_AUDIT_OPINION")
        
        # Disclaimer
        if "disclaimer" in context_lower:
            indicators.append("AUDIT_DISCLAIMER")
        
        return indicators
    
    def _deduplicate(
        self,
        entities: List[PersonEntity]
    ) -> List[PersonEntity]:
        """Deduplicate entities by name."""
        seen: Dict[str, PersonEntity] = {}
        
        for entity in entities:
            key = entity.name.lower()
            
            if key not in seen:
                seen[key] = entity
            else:
                existing = seen[key]
                
                # Merge roles
                for role in entity.roles:
                    if role not in existing.roles:
                        existing.roles.append(role)
                
                # Merge sections
                for section in entity.mentioned_sections:
                    if section not in existing.mentioned_sections:
                        existing.mentioned_sections.append(section)
                
                # Merge fraud indicators
                for indicator in entity.fraud_indicators:
                    if indicator not in existing.fraud_indicators:
                        existing.fraud_indicators.append(indicator)
                
                # Keep higher confidence
                existing.confidence = max(existing.confidence, entity.confidence)
                
                # Prefer EXECUTIVE over BOARD_MEMBER if both found
                if entity.entity_type == "EXECUTIVE":
                    existing.entity_type = "EXECUTIVE"
        
        return list(seen.values())
    
    def find_relationships(
        self,
        entities: List[PersonEntity],
        text: str
    ) -> List[Dict]:
        """
        Find relationships between people entities.
        
        Args:
            entities: List of extracted entities
            text: Original text for context
        
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Look for employment relationships
        for entity in entities:
            if entity.entity_type in ["EXECUTIVE", "KEY_EMPLOYEE", "BOARD_MEMBER"]:
                # Find associated companies
                company_pattern = r"([A-Z][A-Za-z\s\.,&]+(?:Inc|Corp|LLC|Ltd|Co|Company)\.?)"
                
                if entity.context:
                    company_matches = re.findall(company_pattern, entity.context)
                    
                    for company in company_matches[:1]:  # Take first match
                        rel_type = "BOARD_MEMBER_OF" if entity.entity_type == "BOARD_MEMBER" else "WORKS_FOR"
                        relationships.append({
                            "source": entity.name,
                            "target": company.strip(),
                            "relationship_type": rel_type,
                            "properties": {"title": entity.title}
                        })
        
        # Look for family relationships
        family_entities = [e for e in entities if e.entity_type == "FAMILY_RELATION"]
        exec_entities = [e for e in entities if e.entity_type == "EXECUTIVE"]
        
        for family in family_entities:
            for exec_entity in exec_entities:
                # Check if they appear in same context
                if family.context and exec_entity.name.lower() in family.context.lower():
                    relationships.append({
                        "source": family.name,
                        "target": exec_entity.name,
                        "relationship_type": "FAMILY_OF",
                        "properties": {"relationship_type": family.relationship_type}
                    })
        
        return relationships
