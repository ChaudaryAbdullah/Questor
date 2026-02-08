"""
Beneish M-Score Agent
Detects financial statement manipulation using the Beneish M-Score model

The M-Score uses 8 financial ratios to predict earnings manipulation:
M-Score > -2.22 suggests possible manipulation
"""
from typing import Dict, Any, List, Optional
import math

from .base_agent import BaseAgent, AgentResult


class BeneishMScoreAgent(BaseAgent):
    """
    Calculates Beneish M-Score for detecting financial manipulation
    
    Required ratios (can be calculated from financial data):
    - DSRI: Days Sales in Receivables Index
    - GMI: Gross Margin Index
    - AQI: Asset Quality Index
    - SGI: Sales Growth Index
    - DEPI: Depreciation Index
    - SGAI: Sales, General and Administrative Expenses Index
    - LVGI: Leverage Index
    - TATA: Total Accruals to Total Assets
    """
    
    # M-Score coefficients from Beneish (1999)
    COEFFICIENTS = {
        'intercept': -4.84,
        'DSRI': 0.920,
        'GMI': 0.528,
        'AQI': 0.404,
        'SGI': 0.892,
        'DEPI': 0.115,
        'SGAI': -0.172,
        'LVGI': -0.327,
        'TATA': 4.679
    }
    
    THRESHOLD = -2.22  # Values > -2.22 suggest manipulation
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.threshold = self.config.get('threshold', self.THRESHOLD)
    
    def get_name(self) -> str:
        return "beneish_mscore"
    
    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Calculate Beneish M-Score
        
        Args:
            data: Financial data with required ratios or raw financial statements
        
        Returns:
            AgentResult with M-Score and manipulation probability
        """
        # Try to extract or calculate ratios
        ratios = self._extract_ratios(data)
        
        if not ratios:
            # Try to calculate from raw data
            ratios = self._calculate_ratios(data)
        
        if not ratios:
            return self._create_result(
                score=0.0,
                confidence=0.0,
                findings=["Insufficient financial data to calculate M-Score"],
                metrics={},
                success=False,
                error="Missing required financial ratios"
            )
        
        # Calculate M-Score
        m_score = self._calculate_m_score(ratios)
        
        # Convert M-Score to risk score (0-100)
        # M-Score ranges typically from -3 to +3
        # Values > -2.22 are suspicious
        if m_score > self.threshold:
            # Manipulator - scale from threshold to high values
            score = 50 + min(50, (m_score - self.threshold) * 20)
        else:
            # Non-manipulator - scale below threshold
            score = max(0, 50 + (m_score - self.threshold) * 10)
        
        # Confidence based on data completeness
        confidence = len(ratios) / 8.0
        
        # Generate findings
        findings = self._generate_findings(m_score, ratios)
        
        metrics = {
            'm_score': round(m_score, 3),
            'threshold': self.threshold,
            'is_manipulator': m_score > self.threshold,
            'manipulation_probability': round(self._calculate_probability(m_score), 3),
            'ratios': {k: round(v, 3) for k, v in ratios.items()}
        }
        
        return self._create_result(
            score=score,
            confidence=confidence,
            findings=findings,
            metrics=metrics
        )
    
    def _extract_ratios(self, data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract pre-calculated ratios from data"""
        ratios = {}
        
        # Look for ratios in various possible locations
        ratio_data = (
            data.get('beneish_ratios') or
            data.get('financial_ratios') or
            data.get('ratios') or
            {}
        )
        
        for ratio_name in ['DSRI', 'GMI', 'AQI', 'SGI', 'DEPI', 'SGAI', 'LVGI', 'TATA']:
            value = ratio_data.get(ratio_name)
            if value is not None:
                try:
                    ratios[ratio_name] = float(value)
                except (ValueError, TypeError):
                    pass
        
        return ratios if len(ratios) >= 4 else None  # Need at least 4 ratios
    
    def _calculate_ratios(self, data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """
        Calculate ratios from raw financial data
        This is a simplified version - ideally would need 2 years of data
        """
        ratios = {}
        
        # Try to extract financial values
        # This assumes data might have nested financial information
        financial = data.get('financial_data', data)
        
        # Extract common financial metrics if available
        revenue = self._safe_get_float(financial, ['revenue', 'sales', 'total_revenue'])
        receivables = self._safe_get_float(financial, ['receivables', 'accounts_receivable'])
        total_assets = self._safe_get_float(financial, ['total_assets', 'assets'])
        cogs = self._safe_get_float(financial, ['cogs', 'cost_of_goods_sold'])
        
        # Note: Full M-Score calculation requires comparing current vs previous period
        # This is a simplified version using available data
        
        # For demonstration, we'll check if minimal ratios can be approximated
        # In production, you'd need full historical financial statements
        
        return None  # Requires more complete data
    
    def _calculate_m_score(self, ratios: Dict[str, float]) -> float:
        """Calculate M-Score using Beneish formula"""
        m_score = self.COEFFICIENTS['intercept']
        
        for ratio_name, coefficient in self.COEFFICIENTS.items():
            if ratio_name == 'intercept':
                continue
            
            ratio_value = ratios.get(ratio_name, 1.0)  # Default to 1.0 (neutral)
            m_score += coefficient * ratio_value
        
        return m_score
    
    def _calculate_probability(self, m_score: float) -> float:
        """
        Calculate probability of manipulation using logistic function
        P = 1 / (1 + e^(-m_score))
        """
        try:
            probability = 1 / (1 + math.exp(-m_score))
            return probability
        except:
            return 0.5
    
    def _generate_findings(self, m_score: float, ratios: Dict[str, float]) -> List[str]:
        """Generate findings based on M-Score and ratios"""
        findings = []
        
        if m_score > self.threshold:
            findings.append(f"M-Score ({m_score:.2f}) exceeds threshold ({self.threshold}) - possible manipulation")
            
            # Identify problematic ratios
            if ratios.get('DSRI', 1.0) > 1.5:
                findings.append("High DSRI: Receivables growing faster than sales")
            if ratios.get('GMI', 1.0) > 1.2:
                findings.append("High GMI: Gross margin deterioration")
            if ratios.get('AQI', 1.0) > 1.5:
                findings.append("High AQI: Increased asset risk")
            if ratios.get('SGI', 1.0) > 1.5:
                findings.append("High SGI: Aggressive sales growth")
            if ratios.get('TATA', 0.0) > 0.03:
                findings.append("High TATA: Excessive accruals")
        else:
            findings.append(f"M-Score ({m_score:.2f}) below threshold - no manipulation detected")
        
        return findings
    
    def _safe_get_float(self, data: Dict, keys: List[str]) -> Optional[float]:
        """Safely extract float from multiple possible keys"""
        for key in keys:
            value = data.get(key)
            if value is not None:
                try:
                    return float(value)
                except (ValueError, TypeError):
                    pass
        return None
    
    def is_applicable(self, data: Dict[str, Any]) -> bool:
        """Check if data contains financial ratios or raw financial data"""
        # Check for ratios
        if self._extract_ratios(data):
            return True
        
        # Check for raw financial data
        financial = data.get('financial_data', data)
        required_fields = ['revenue', 'receivables', 'total_assets']
        
        return any(field in financial for field in required_fields)
