"""
Benford's Law Agent
Detects anomalies in numerical distributions using Benford's Law

Benford's Law states that in many naturally occurring datasets, the leading 
digit is more likely to be small. First digit probabilities:
1: 30.1%, 2: 17.6%, 3: 12.5%, 4: 9.7%, 5: 7.9%, 6: 6.7%, 7: 5.8%, 8: 5.1%, 9: 4.6%
"""
import math
from typing import Dict, Any, List, Tuple
from collections import Counter
import json

from .base_agent import BaseAgent, AgentResult


class BenfordsLawAgent(BaseAgent):
    """
    Analyzes numerical data for conformance to Benford's Law
    """
    
    # Expected frequencies for first digit (Benford's Law)
    EXPECTED_FREQ = {
        1: 0.301,
        2: 0.176,
        3: 0.125,
        4: 0.097,
        5: 0.079,
        6: 0.067,
        7: 0.058,
        8: 0.051,
        9: 0.046
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.min_samples = self.config.get('min_samples', 30)
        self.critical_chi_square = self.config.get('critical_chi_square', 15.507)  # 95% confidence, 8 df
    
    def get_name(self) -> str:
        return "benfords_law"
    
    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Analyze numerical fields for Benford's Law compliance
        
        Args:
            data: Structured data with numerical fields
        
        Returns:
            AgentResult with deviation score
        """
        # Extract numerical values from data
        numbers = self._extract_numbers(data)
        
        if len(numbers) < self.min_samples:
            return self._create_result(
                score=0.0,
                confidence=0.0,
                findings=[f"Insufficient data: {len(numbers)} numbers (minimum {self.min_samples})"],
                metrics={'sample_count': len(numbers)},
                success=False,
                error=f"Insufficient samples"
            )
        
        # Get first digit distribution
        first_digits = [self._get_first_digit(n) for n in numbers]
        first_digits = [d for d in first_digits if d is not None]
        
        if not first_digits:
            return self._create_result(
                score=0.0,
                confidence=0.0,
                findings=["No valid numerical data found"],
                metrics={},
                success=False,
                error="No valid numbers"
            )
        
        # Calculate observed frequencies
        digit_counts = Counter(first_digits)
        total = len(first_digits)
        observed_freq = {d: digit_counts.get(d, 0) / total for d in range(1, 10)}
        
        # Calculate chi-square statistic
        chi_square = self._calculate_chi_square(observed_freq, total)
        
        # Calculate deviation score (0-100)
        # Higher chi-square = more deviation = higher suspicion
        score = min(100.0, (chi_square / self.critical_chi_square) * 50)
        
        # Confidence based on sample size
        confidence = min(1.0, total / 100.0)
        
        # Identify anomalous digits
        findings = self._identify_anomalies(observed_freq)
        
        metrics = {
            'chi_square': round(chi_square, 3),
            'sample_count': total,
            'observed_frequencies': {str(k): round(v, 3) for k, v in observed_freq.items()},
            'expected_frequencies': {str(k): round(v, 3) for k, v in self.EXPECTED_FREQ.items()},
            'passes_test': chi_square < self.critical_chi_square
        }
        
        return self._create_result(
            score=score,
            confidence=confidence,
            findings=findings,
            metrics=metrics
        )
    
    def _extract_numbers(self, data: Dict[str, Any]) -> List[float]:
        """Extract all numerical values from data"""
        numbers = []
        
        def extract_recursive(obj):
            if isinstance(obj, (int, float)) and obj > 0:
                numbers.append(abs(float(obj)))
            elif isinstance(obj, dict):
                for value in obj.values():
                    extract_recursive(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_recursive(item)
        
        extract_recursive(data)
        return numbers
    
    def _get_first_digit(self, number: float) -> int:
        """Extract first significant digit"""
        if number <= 0:
            return None
        
        # Convert to string and find first non-zero digit
        num_str = f"{number:.10f}".replace('.', '').lstrip('0')
        if num_str:
            return int(num_str[0])
        return None
    
    def _calculate_chi_square(self, observed_freq: Dict[int, float], total: int) -> float:
        """Calculate chi-square statistic"""
        chi_square = 0.0
        
        for digit in range(1, 10):
            expected_count = self.EXPECTED_FREQ[digit] * total
            observed_count = observed_freq.get(digit, 0) * total
            
            if expected_count > 0:
                chi_square += ((observed_count - expected_count) ** 2) / expected_count
        
        return chi_square
    
    def _identify_anomalies(self, observed_freq: Dict[int, float]) -> List[str]:
        """Identify digits with significant deviations"""
        findings = []
        
        for digit in range(1, 10):
            expected = self.EXPECTED_FREQ[digit]
            observed = observed_freq.get(digit, 0)
            deviation = abs(observed - expected) / expected
            
            if deviation > 0.3:  # >30% deviation
                direction = "over" if observed > expected else "under"
                findings.append(
                    f"Digit {digit}: {direction}-represented "
                    f"(observed: {observed*100:.1f}%, expected: {expected*100:.1f}%)"
                )
        
        if not findings:
            findings.append("Distribution conforms to Benford's Law")
        
        return findings
    
    def is_applicable(self, data: Dict[str, Any]) -> bool:
        """Check if data contains enough numerical values"""
        numbers = self._extract_numbers(data)
        return len(numbers) >= self.min_samples
