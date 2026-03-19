"""
Temporal Entity Extractor
Specializes in extracting dates, fiscal periods, and event timelines.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from ..utils.logger import get_nlp_logger


@dataclass
class TemporalEntity:
    """Represents a temporal entity."""
    text: str
    entity_type: str  # DATE, FISCAL_PERIOD, FISCAL_QUARTER, EVENT_DATE
    normalized_date: Optional[str] = None
    year: Optional[int] = None
    quarter: Optional[int] = None
    event_type: Optional[str] = None
    section: str = ""
    context: str = ""
    fraud_indicators: List[str] = field(default_factory=list)
    confidence: float = 0.0


class TemporalExtractor:
    """Extractor for temporal entities."""
    
    MONTHS = {
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
        "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6,
        "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
    }
    
    DATE_PATTERNS = [
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})",
        r"(\d{1,2})/(\d{1,2})/(\d{2,4})",
        r"(\d{4})-(\d{2})-(\d{2})",
    ]
    
    FISCAL_PATTERNS = [
        r"fiscal\s+(?:year\s+)?(\d{4})",
        r"FY\s*['\"]?(\d{2,4})",
        r"(?:year|fiscal\s+year)\s+ended?\s+(\w+\s+\d{1,2},?\s+\d{4})",
        r"(?:for\s+the\s+)?(?:twelve|twelvemonth)\s+months?\s+ended?\s+(\w+\s+\d{1,2},?\s+\d{4})",
    ]
    
    QUARTER_PATTERNS = [
        r"Q([1-4])\s+(\d{4})",
        r"(first|second|third|fourth)\s+quarter\s+(?:of\s+)?(\d{4})",
        r"(?:three|3)\s+months?\s+ended?\s+(\w+\s+\d{1,2},?\s+\d{4})",
    ]
    
    EVENT_PATTERNS = [
        (r"(?:announced|completed|entered\s+into|signed)\s+(?:on\s+)?(\w+\s+\d{1,2},?\s+\d{4})", "announcement"),
        (r"effective\s+(?:as\s+of\s+)?(\w+\s+\d{1,2},?\s+\d{4})", "effective_date"),
        (r"(?:acquisition|merger)\s+(?:on|of)\s+(\w+\s+\d{1,2},?\s+\d{4})", "acquisition"),
        (r"(?:IPO|initial\s+public\s+offering)\s+(?:on\s+)?(\w+\s+\d{1,2},?\s+\d{4})", "ipo"),
    ]
    
    def __init__(self):
        self.logger = get_nlp_logger()
        self.date_patterns = [re.compile(p, re.IGNORECASE) for p in self.DATE_PATTERNS]
        self.fiscal_patterns = [re.compile(p, re.IGNORECASE) for p in self.FISCAL_PATTERNS]
        self.quarter_patterns = [re.compile(p, re.IGNORECASE) for p in self.QUARTER_PATTERNS]
        self.event_patterns = [(re.compile(p, re.IGNORECASE), t) for p, t in self.EVENT_PATTERNS]
    
    def extract(self, text: str, section: str = "") -> List[TemporalEntity]:
        """Extract temporal entities from text."""
        entities = []
        entities.extend(self._extract_dates(text, section))
        entities.extend(self._extract_fiscal_periods(text, section))
        entities.extend(self._extract_quarters(text, section))
        entities.extend(self._extract_events(text, section))
        return self._deduplicate(entities)
    
    def _extract_dates(self, text: str, section: str) -> List[TemporalEntity]:
        """Extract date entities."""
        entities = []
        for pattern in self.date_patterns:
            for match in pattern.finditer(text):
                normalized = self._normalize_date(match.group(0))
                context = self._get_context(text, match.start(), match.end())
                entities.append(TemporalEntity(
                    text=match.group(0), entity_type="DATE",
                    normalized_date=normalized, section=section,
                    context=context, confidence=0.9
                ))
        return entities
    
    def _extract_fiscal_periods(self, text: str, section: str) -> List[TemporalEntity]:
        """Extract fiscal period entities."""
        entities = []
        for pattern in self.fiscal_patterns:
            for match in pattern.finditer(text):
                year = self._extract_year(match.group(0))
                context = self._get_context(text, match.start(), match.end())
                fraud_indicators = self._check_period_fraud_indicators(context)
                entities.append(TemporalEntity(
                    text=match.group(0), entity_type="FISCAL_PERIOD",
                    year=year, section=section, context=context,
                    fraud_indicators=fraud_indicators, confidence=0.85
                ))
        return entities
    
    def _extract_quarters(self, text: str, section: str) -> List[TemporalEntity]:
        """Extract quarter entities."""
        entities = []
        quarter_map = {"first": 1, "second": 2, "third": 3, "fourth": 4}
        for pattern in self.quarter_patterns:
            for match in pattern.finditer(text):
                groups = match.groups()
                quarter = None
                year = None
                for g in groups:
                    if g:
                        if g.lower() in quarter_map:
                            quarter = quarter_map[g.lower()]
                        elif g.isdigit() and len(g) <= 1:
                            quarter = int(g)
                        elif g.isdigit() and len(g) == 4:
                            year = int(g)
                context = self._get_context(text, match.start(), match.end())
                entities.append(TemporalEntity(
                    text=match.group(0), entity_type="FISCAL_QUARTER",
                    year=year, quarter=quarter, section=section,
                    context=context, confidence=0.85
                ))
        return entities
    
    def _extract_events(self, text: str, section: str) -> List[TemporalEntity]:
        """Extract event date entities."""
        entities = []
        for pattern, event_type in self.event_patterns:
            for match in pattern.finditer(text):
                normalized = self._normalize_date(match.group(1) if match.groups() else match.group(0))
                context = self._get_context(text, match.start(), match.end())
                entities.append(TemporalEntity(
                    text=match.group(0), entity_type="EVENT_DATE",
                    normalized_date=normalized, event_type=event_type,
                    section=section, context=context, confidence=0.8
                ))
        return entities
    
    def _normalize_date(self, date_str: str) -> Optional[str]:
        """Normalize date to ISO format."""
        for pattern in self.date_patterns:
            match = pattern.match(date_str)
            if match:
                try:
                    groups = match.groups()
                    if len(groups) == 3:
                        if groups[0].lower() in self.MONTHS:
                            month = self.MONTHS[groups[0].lower()]
                            day = int(groups[1])
                            year = int(groups[2])
                        else:
                            month = int(groups[0])
                            day = int(groups[1])
                            year = int(groups[2])
                            if year < 100: year += 2000
                        return f"{year:04d}-{month:02d}-{day:02d}"
                except (ValueError, IndexError):
                    pass
        return None
    
    def _extract_year(self, text: str) -> Optional[int]:
        """Extract year from text."""
        match = re.search(r'\b(19|20)\d{2}\b', text)
        if match:
            return int(match.group())
        match = re.search(r'\b\d{2}\b', text)
        if match:
            year = int(match.group())
            return 2000 + year if year < 50 else 1900 + year
        return None
    
    def _get_context(self, text: str, start: int, end: int, window: int = 100) -> str:
        return text[max(0, start - window):min(len(text), end + window)].strip()
    
    def _check_period_fraud_indicators(self, context: str) -> List[str]:
        """Check for period-related fraud indicators."""
        indicators = []
        context_lower = context.lower()
        if "restat" in context_lower: indicators.append("RESTATEMENT")
        if "prior period" in context_lower and "adjustment" in context_lower:
            indicators.append("PRIOR_PERIOD_ADJUSTMENT")
        if "changed" in context_lower and "policy" in context_lower:
            indicators.append("POLICY_CHANGE")
        return indicators
    
    def _deduplicate(self, entities: List[TemporalEntity]) -> List[TemporalEntity]:
        """Deduplicate entities."""
        seen = {}
        for entity in entities:
            key = f"{entity.entity_type}:{entity.text}"
            if key not in seen:
                seen[key] = entity
        return list(seen.values())
