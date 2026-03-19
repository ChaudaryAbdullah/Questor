"""
Fraud Pattern Detector
Implements known fraud pattern detection logic.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

from ..utils.logger import get_validation_logger


@dataclass
class FraudFinding:
    """Represents a fraud finding."""
    pattern_name: str
    severity: str
    description: str
    entities_involved: List[str]
    evidence: List[str]
    confidence: float
    sec_rule_violated: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


class FraudPatternDetector:
    """Detects known fraud patterns in extracted data."""
    
    def __init__(self):
        self.logger = get_validation_logger()
    
    def detect_undisclosed_relationships(
        self,
        entities: List[Dict],
        relationships: List[Dict],
        sections: List[str]
    ) -> List[FraudFinding]:
        """Detect undisclosed related party relationships."""
        findings = []
        
        # Find related parties
        related_parties = [e for e in entities if e.get("entity_type") == "RELATED_PARTY"]
        
        for rp in related_parties:
            # Check if disclosed in footnotes
            mentioned_sections = rp.get("mentioned_sections", [])
            in_footnotes = any("footnote" in s.lower() or "note" in s.lower() 
                             for s in mentioned_sections)
            
            if not in_footnotes:
                findings.append(FraudFinding(
                    pattern_name="UNDISCLOSED_RELATED_PARTY",
                    severity="HIGH",
                    description=f"Related party '{rp.get('text', 'Unknown')}' may not be properly disclosed in footnotes",
                    entities_involved=[rp.get("id", "")],
                    evidence=[f"Mentioned in: {', '.join(mentioned_sections)}"],
                    confidence=0.7,
                    sec_rule_violated="S-K Item 404",
                    recommendations=["Verify related party disclosure in footnotes"]
                ))
        
        return findings
    
    def detect_subsidiary_hiding(
        self,
        entities: List[Dict],
        sections: List[str]
    ) -> List[FraudFinding]:
        """Detect hidden subsidiaries."""
        findings = []
        
        subsidiaries = [e for e in entities if e.get("entity_type") == "SUBSIDIARY"]
        
        for sub in subsidiaries:
            mentioned_sections = sub.get("mentioned_sections", [])
            in_footnotes = any("footnote" in s.lower() or "consolidation" in s.lower() 
                             for s in mentioned_sections)
            
            if not in_footnotes:
                findings.append(FraudFinding(
                    pattern_name="SUBSIDIARY_HIDING",
                    severity="CRITICAL",
                    description=f"Subsidiary '{sub.get('text', 'Unknown')}' mentioned but not in consolidation footnotes",
                    entities_involved=[sub.get("id", "")],
                    evidence=[f"Mentioned in: {', '.join(mentioned_sections)}"],
                    confidence=0.8,
                    sec_rule_violated="S-X 4-08(g)",
                    recommendations=["Review consolidation disclosures"]
                ))
        
        return findings
    
    def detect_circular_transactions(
        self,
        graph_builder
    ) -> List[FraudFinding]:
        """Detect circular transaction patterns."""
        findings = []
        
        try:
            cycles = graph_builder.detect_cycles()
            
            for cycle in cycles:
                if len(cycle) > 2:
                    findings.append(FraudFinding(
                        pattern_name="CIRCULAR_TRANSACTIONS",
                        severity="CRITICAL",
                        description=f"Circular transaction pattern detected involving {len(cycle)} entities",
                        entities_involved=cycle,
                        evidence=[f"Transaction cycle: {' -> '.join(cycle)}"],
                        confidence=0.9,
                        recommendations=["Investigate transaction legitimacy"]
                    ))
        except Exception as e:
            self.logger.warning(f"Cycle detection failed: {e}")
        
        return findings
    
    def detect_all_patterns(
        self,
        entities: List[Dict],
        relationships: List[Dict],
        sections: List[str],
        graph_builder = None
    ) -> List[FraudFinding]:
        """Run all fraud pattern detections."""
        all_findings = []
        
        all_findings.extend(self.detect_undisclosed_relationships(entities, relationships, sections))
        all_findings.extend(self.detect_subsidiary_hiding(entities, sections))
        
        if graph_builder:
            all_findings.extend(self.detect_circular_transactions(graph_builder))
        
        self.logger.info(f"Detected {len(all_findings)} potential fraud indicators")
        return all_findings
