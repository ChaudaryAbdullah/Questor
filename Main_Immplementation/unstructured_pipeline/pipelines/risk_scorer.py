# File: pipelines/risk_scorer.py
"""
Risk Scoring Module for Financial Fraud Detection
Calculates risk scores based on extracted entities, relationships, and patterns
"""
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re
from datetime import datetime

try:
    from utils import Config, Logger
except ImportError:
    from utils.config import Config
    from utils.logger import Logger


class RiskScorer:
    """
    Calculates fraud risk scores for documents based on:
    1. Fraud indicators (keywords and patterns)
    2. Entity relationships (suspicious connections)
    3. Financial anomalies (unusual amounts, patterns)
    4. Temporal patterns (timing of events)
    """
    
    # Risk weights for different components
    WEIGHTS = {
        'fraud_indicators': 0.35,
        'entity_risk': 0.25,
        'financial_anomalies': 0.25,
        'relationship_risk': 0.15
    }
    
    # High-risk fraud indicators with severity scores
    FRAUD_INDICATORS = {
        # Severe indicators (0.8-1.0)
        'fictitious': 0.95,
        'fabricated': 0.95,
        'concealment': 0.90,
        'material weakness': 0.90,
        'restatement': 0.85,
        'manipulation': 0.85,
        'misstatement': 0.80,
        
        # High indicators (0.6-0.8)
        'overstatement': 0.75,
        'understatement': 0.75,
        'round-trip': 0.70,
        'special purpose entity': 0.70,
        'off-balance sheet': 0.65,
        'related party': 0.60,
        
        # Medium indicators (0.4-0.6)
        'derivative': 0.55,
        'restructuring': 0.50,
        'write-off': 0.50,
        'write-down': 0.50,
        'impairment': 0.45,
        'goodwill': 0.40,
        
        # Low indicators (0.2-0.4)
        'contingency': 0.35,
        'subsidiary': 0.30,
        'joint venture': 0.25,
        'revenue recognition': 0.20
    }
    
    # Suspicious relationship patterns
    SUSPICIOUS_RELATIONSHIPS = {
        'CONCEALED': 0.90,
        'TRANSFERRED': 0.70,
        'CREATED': 0.50,
        'OWNS': 0.40,
        'CONTROLS': 0.40,
        'MANAGES': 0.30
    }
    
    def __init__(self):
        """Initialize the risk scorer"""
        self.logger = Logger.get_logger(self.__class__.__name__)
        self.logger.info("Risk Scorer initialized")
    
    def calculate_document_risk(
        self,
        document: Dict[str, Any],
        entities: Optional[Dict[str, List[Dict]]] = None,
        relationships: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score for a document
        
        Args:
            document: Document data with content and metadata
            entities: Extracted entities (optional)
            relationships: Extracted relationships (optional)
            
        Returns:
            Dictionary containing risk score and detailed breakdown
        """
        content = document.get('content', '')
        doc_id = document.get('doc_id', 'unknown')
        
        # Calculate individual risk components
        fraud_indicator_score = self._calculate_fraud_indicator_score(content, entities)
        entity_risk_score = self._calculate_entity_risk_score(entities)
        financial_anomaly_score = self._calculate_financial_anomaly_score(content, entities)
        relationship_risk_score = self._calculate_relationship_risk_score(relationships)
        
        # Calculate weighted overall risk score (0-100)
        overall_risk = (
            fraud_indicator_score * self.WEIGHTS['fraud_indicators'] +
            entity_risk_score * self.WEIGHTS['entity_risk'] +
            financial_anomaly_score * self.WEIGHTS['financial_anomalies'] +
            relationship_risk_score * self.WEIGHTS['relationship_risk']
        ) * 100
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_risk)
        
        # Extract key risk factors
        risk_factors = self._extract_risk_factors(
            content, entities, relationships,
            fraud_indicator_score, entity_risk_score,
            financial_anomaly_score, relationship_risk_score
        )
        
        risk_data = {
            'doc_id': doc_id,
            'overall_risk_score': round(overall_risk, 2),
            'risk_level': risk_level,
            'component_scores': {
                'fraud_indicators': round(fraud_indicator_score * 100, 2),
                'entity_risk': round(entity_risk_score * 100, 2),
                'financial_anomalies': round(financial_anomaly_score * 100, 2),
                'relationship_risk': round(relationship_risk_score * 100, 2)
            },
            'risk_factors': risk_factors,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.debug(f"Document {doc_id} risk score: {overall_risk:.2f} ({risk_level})")
        
        return risk_data
    
    def _calculate_fraud_indicator_score(
        self,
        content: str,
        entities: Optional[Dict[str, List[Dict]]] = None
    ) -> float:
        """
        Calculate score based on fraud indicator keywords
        
        Returns:
            Score between 0.0 and 1.0
        """
        content_lower = content.lower()
        total_score = 0.0
        indicator_count = 0
        
        # Check for fraud indicator keywords
        for indicator, severity in self.FRAUD_INDICATORS.items():
            count = content_lower.count(indicator)
            if count > 0:
                # Logarithmic scaling to prevent single indicator from dominating
                indicator_score = severity * min(1.0, 0.3 + 0.1 * count)
                total_score += indicator_score
                indicator_count += 1
        
        # Check entities for fraud indicators
        if entities and 'FRAUD_INDICATOR' in entities:
            fraud_entity_count = len(entities['FRAUD_INDICATOR'])
            total_score += min(0.5, fraud_entity_count * 0.1)
            indicator_count += fraud_entity_count
        
        # Normalize score
        if indicator_count > 0:
            # Average score with diminishing returns
            normalized_score = min(1.0, total_score / max(1, indicator_count * 0.8))
        else:
            normalized_score = 0.0
        
        return normalized_score
    
    def _calculate_entity_risk_score(
        self,
        entities: Optional[Dict[str, List[Dict]]] = None
    ) -> float:
        """
        Calculate risk based on entity types and patterns
        
        Returns:
            Score between 0.0 and 1.0
        """
        if not entities:
            return 0.0
        
        risk_score = 0.0
        
        # Count high-risk entity types
        financial_terms = len(entities.get('FINANCIAL_TERM', []))
        fraud_indicators = len(entities.get('FRAUD_INDICATOR', []))
        organizations = len(entities.get('ORG', []))
        money_entities = len(entities.get('MONEY', []))
        
        # Score based on entity density
        if financial_terms > 5:
            risk_score += min(0.4, financial_terms * 0.05)
        
        if fraud_indicators > 0:
            risk_score += min(0.5, fraud_indicators * 0.15)
        
        # Multiple organizations can indicate complex structures
        if organizations > 3:
            risk_score += min(0.3, (organizations - 3) * 0.05)
        
        # Multiple monetary amounts can indicate complex transactions
        if money_entities > 3:
            risk_score += min(0.2, (money_entities - 3) * 0.03)
        
        return min(1.0, risk_score)
    
    def _calculate_financial_anomaly_score(
        self,
        content: str,
        entities: Optional[Dict[str, List[Dict]]] = None
    ) -> float:
        """
        Calculate score based on financial anomalies and patterns
        
        Returns:
            Score between 0.0 and 1.0
        """
        risk_score = 0.0
        
        # Extract monetary amounts
        money_pattern = r'\$\s*(\d+(?:\.\d+)?)\s*(million|billion|thousand|M|B|K)?'
        amounts = re.findall(money_pattern, content, re.IGNORECASE)
        
        if amounts:
            # Convert to numerical values
            numerical_amounts = []
            for amount, unit in amounts:
                value = float(amount)
                unit_lower = unit.lower() if unit else ''
                
                if unit_lower in ['million', 'm']:
                    value *= 1_000_000
                elif unit_lower in ['billion', 'b']:
                    value *= 1_000_000_000
                elif unit_lower in ['thousand', 'k']:
                    value *= 1_000
                
                numerical_amounts.append(value)
            
            # Check for large amounts (potential risk)
            if numerical_amounts:
                max_amount = max(numerical_amounts)
                if max_amount > 100_000_000:  # > $100M
                    risk_score += 0.3
                elif max_amount > 10_000_000:  # > $10M
                    risk_score += 0.2
                elif max_amount > 1_000_000:  # > $1M
                    risk_score += 0.1
            
            # Check for many different amounts (complex transactions)
            if len(set(numerical_amounts)) > 5:
                risk_score += 0.2
        
        # Check for percentage mentions (often in fraud contexts)
        percentage_pattern = r'\d+(?:\.\d+)?%'
        percentages = re.findall(percentage_pattern, content)
        if len(percentages) > 3:
            risk_score += 0.1
        
        # Check for date patterns (multiple dates can indicate timeline manipulation)
        date_pattern = r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, content, re.IGNORECASE)
        if len(dates) > 5:
            risk_score += 0.15
        
        return min(1.0, risk_score)
    
    def _calculate_relationship_risk_score(
        self,
        relationships: Optional[List[Dict]] = None
    ) -> float:
        """
        Calculate risk based on entity relationships
        
        Returns:
            Score between 0.0 and 1.0
        """
        if not relationships:
            return 0.0
        
        risk_score = 0.0
        
        for rel in relationships:
            relation_type = rel.get('relation_type', 'RELATED_TO')
            
            # Check if it's a suspicious relationship type
            if relation_type in self.SUSPICIOUS_RELATIONSHIPS:
                risk_score += self.SUSPICIOUS_RELATIONSHIPS[relation_type]
            
            # Check for fraud-related predicates
            predicate = rel.get('predicate', '').lower()
            if predicate in ['concealed', 'hid', 'transferred', 'manipulated']:
                risk_score += 0.3
        
        # Normalize by number of relationships (with diminishing returns)
        if relationships:
            normalized_score = min(1.0, risk_score / max(1, len(relationships) * 0.5))
        else:
            normalized_score = 0.0
        
        return normalized_score
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine categorical risk level from numerical score
        
        Args:
            risk_score: Risk score (0-100)
            
        Returns:
            Risk level: CRITICAL, HIGH, MEDIUM, LOW, or MINIMAL
        """
        if risk_score >= 80:
            return 'CRITICAL'
        elif risk_score >= 60:
            return 'HIGH'
        elif risk_score >= 40:
            return 'MEDIUM'
        elif risk_score >= 20:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _extract_risk_factors(
        self,
        content: str,
        entities: Optional[Dict],
        relationships: Optional[List[Dict]],
        fraud_score: float,
        entity_score: float,
        financial_score: float,
        relationship_score: float
    ) -> List[str]:
        """
        Extract specific risk factors found in the document
        
        Returns:
            List of human-readable risk factors
        """
        risk_factors = []
        
        # Fraud indicators
        if fraud_score > 0.5:
            content_lower = content.lower()
            found_indicators = [
                indicator for indicator in self.FRAUD_INDICATORS.keys()
                if indicator in content_lower
            ]
            if found_indicators:
                risk_factors.append(f"High-risk keywords detected: {', '.join(found_indicators[:3])}")
        
        # Entity-based risks
        if entities:
            if entity_score > 0.4:
                fraud_indicators = entities.get('FRAUD_INDICATOR', [])
                if fraud_indicators:
                    risk_factors.append(f"Fraud indicators found: {len(fraud_indicators)} instances")
                
                financial_terms = entities.get('FINANCIAL_TERM', [])
                if len(financial_terms) > 5:
                    risk_factors.append(f"Complex financial structures: {len(financial_terms)} terms")
                
                orgs = entities.get('ORG', [])
                if len(orgs) > 3:
                    risk_factors.append(f"Multiple organizations involved: {len(orgs)} entities")
        
        # Financial anomalies
        if financial_score > 0.3:
            money_pattern = r'\$\s*\d+(?:\.\d+)?(?:\s*(?:million|billion|M|B))?'
            amounts = re.findall(money_pattern, content, re.IGNORECASE)
            if amounts:
                risk_factors.append(f"Large financial amounts: {len(amounts)} transactions")
        
        # Relationship risks
        if relationships and relationship_score > 0.3:
            suspicious_rels = [
                rel for rel in relationships
                if rel.get('relation_type') in self.SUSPICIOUS_RELATIONSHIPS
            ]
            if suspicious_rels:
                risk_factors.append(f"Suspicious relationships: {len(suspicious_rels)} connections")
        
        # If no specific factors, add general assessment
        if not risk_factors:
            risk_factors.append("No significant risk factors detected")
        
        return risk_factors
    
    def calculate_batch_risk_scores(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Calculate risk scores for multiple documents
        
        Args:
            documents: List of document dictionaries with entities and relationships
            
        Returns:
            List of risk score dictionaries
        """
        risk_scores = []
        
        for doc in documents:
            try:
                risk_data = self.calculate_document_risk(
                    document=doc,
                    entities=doc.get('entities'),
                    relationships=doc.get('relationships')
                )
                risk_scores.append(risk_data)
            except Exception as e:
                self.logger.error(f"Failed to calculate risk for doc {doc.get('doc_id')}: {str(e)}")
                # Add minimal risk data for failed documents
                risk_scores.append({
                    'doc_id': doc.get('doc_id', 'unknown'),
                    'overall_risk_score': 0.0,
                    'risk_level': 'UNKNOWN',
                    'error': str(e)
                })
        
        return risk_scores
