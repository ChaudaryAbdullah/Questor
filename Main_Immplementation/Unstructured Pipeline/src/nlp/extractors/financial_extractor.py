"""
Financial Entity Extractor
Specializes in extracting revenue, expenses, assets, liabilities.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from ..utils.logger import get_nlp_logger


@dataclass
class FinancialEntity:
    """Represents a financial entity."""
    text: str
    entity_type: str
    sub_type: Optional[str] = None
    amount: Optional[float] = None
    currency: str = "USD"
    unit: Optional[str] = None
    section: str = ""
    context: str = ""
    fraud_indicators: List[str] = field(default_factory=list)
    confidence: float = 0.0


class FinancialExtractor:
    """Extractor for financial entities."""
    
    MONEY_PATTERN = r"\$\s*([\d,]+(?:\.\d+)?)\s*(million|billion|thousand|M|B|K)?"
    
    REVENUE_PATTERNS = [
        r"(?:total\s+)?(?:net\s+)?revenue[s]?\s+(?:of\s+)?(\$[\d,\.]+\s*(?:million|billion|M|B)?)",
        r"(?:total\s+)?(?:net\s+)?sales\s+(?:of\s+)?(\$[\d,\.]+\s*(?:million|billion|M|B)?)",
    ]
    
    EXPENSE_PATTERNS = [
        r"(?:operating\s+)?expense[s]?\s+(?:of\s+)?(\$[\d,\.]+\s*(?:million|billion|M|B)?)",
        r"cost\s+of\s+(?:revenue|sales)\s+(?:was\s+)?(\$[\d,\.]+\s*(?:million|billion|M|B)?)",
    ]
    
    ASSET_PATTERNS = [
        r"total\s+assets?\s+(?:of\s+)?(\$[\d,\.]+\s*(?:million|billion|M|B)?)",
        r"goodwill\s+(?:of\s+)?(\$[\d,\.]+\s*(?:million|billion|M|B)?)",
    ]
    
    LIABILITY_PATTERNS = [
        r"total\s+liabilities?\s+(?:of\s+)?(\$[\d,\.]+\s*(?:million|billion|M|B)?)",
        r"(?:long-term\s+)?debt\s+(?:of\s+)?(\$[\d,\.]+\s*(?:million|billion|M|B)?)",
    ]
    
    def __init__(self):
        self.logger = get_nlp_logger()
        self.money_regex = re.compile(self.MONEY_PATTERN, re.IGNORECASE)
        self.revenue_patterns = [re.compile(p, re.IGNORECASE) for p in self.REVENUE_PATTERNS]
        self.expense_patterns = [re.compile(p, re.IGNORECASE) for p in self.EXPENSE_PATTERNS]
        self.asset_patterns = [re.compile(p, re.IGNORECASE) for p in self.ASSET_PATTERNS]
        self.liability_patterns = [re.compile(p, re.IGNORECASE) for p in self.LIABILITY_PATTERNS]
    
    def extract(self, text: str, section: str = "") -> List[FinancialEntity]:
        """Extract financial entities from text."""
        entities = []
        entities.extend(self._extract_monetary_values(text, section))
        entities.extend(self._extract_typed(text, section, self.revenue_patterns, "REVENUE"))
        entities.extend(self._extract_typed(text, section, self.expense_patterns, "EXPENSE"))
        entities.extend(self._extract_typed(text, section, self.asset_patterns, "ASSET"))
        entities.extend(self._extract_typed(text, section, self.liability_patterns, "LIABILITY"))
        return self._deduplicate(entities)
    
    def _extract_monetary_values(self, text: str, section: str) -> List[FinancialEntity]:
        """Extract all monetary values."""
        entities = []
        for match in self.money_regex.finditer(text):
            amount_str = match.group(1).replace(",", "")
            unit = match.group(2)
            try:
                amount = float(amount_str)
                if unit:
                    unit_lower = unit.lower()
                    if unit_lower in ["million", "m"]: amount *= 1_000_000
                    elif unit_lower in ["billion", "b"]: amount *= 1_000_000_000
                    elif unit_lower in ["thousand", "k"]: amount *= 1_000
                context = self._get_context(text, match.start(), match.end())
                entities.append(FinancialEntity(
                    text=match.group(0), entity_type="MONETARY_VALUE",
                    amount=amount, currency="USD", unit=unit,
                    section=section, context=context, confidence=0.9
                ))
            except ValueError:
                pass
        return entities
    
    def _extract_typed(self, text: str, section: str, patterns: List, entity_type: str) -> List[FinancialEntity]:
        """Extract typed financial entities."""
        entities = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                context = self._get_context(text, match.start(), match.end())
                amount = self._parse_amount(match.group(1) if match.groups() else match.group(0))
                fraud_indicators = self._check_fraud_indicators(context, entity_type)
                entities.append(FinancialEntity(
                    text=match.group(0), entity_type=entity_type,
                    amount=amount, section=section, context=context,
                    fraud_indicators=fraud_indicators, confidence=0.85
                ))
        return entities
    
    def _parse_amount(self, amount_str: str) -> Optional[float]:
        """Parse amount string to float."""
        if not amount_str: return None
        match = re.search(r'[\d,\.]+', amount_str)
        if not match: return None
        try:
            amount = float(match.group().replace(",", ""))
            amount_lower = amount_str.lower()
            if "billion" in amount_lower: amount *= 1_000_000_000
            elif "million" in amount_lower: amount *= 1_000_000
            return amount
        except ValueError:
            return None
    
    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        return text[max(0, start - window):min(len(text), end + window)].strip()
    
    def _check_fraud_indicators(self, context: str, entity_type: str) -> List[str]:
        """Check for fraud indicators."""
        indicators = []
        context_lower = context.lower()
        if entity_type == "REVENUE":
            if "related party" in context_lower: indicators.append("RELATED_PARTY_REVENUE")
            if "bill and hold" in context_lower: indicators.append("BILL_AND_HOLD")
        elif entity_type == "LIABILITY":
            if "off-balance" in context_lower: indicators.append("OFF_BALANCE_SHEET")
            if "contingent" in context_lower: indicators.append("CONTINGENT_LIABILITY")
        return indicators
    
    def _deduplicate(self, entities: List[FinancialEntity]) -> List[FinancialEntity]:
        """Deduplicate entities."""
        seen = {}
        for entity in entities:
            key = f"{entity.entity_type}:{entity.amount}:{entity.sub_type}"
            if key not in seen:
                seen[key] = entity
            else:
                existing = seen[key]
                for ind in entity.fraud_indicators:
                    if ind not in existing.fraud_indicators:
                        existing.fraud_indicators.append(ind)
        return list(seen.values())
