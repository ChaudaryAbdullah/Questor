"""
Constants for the Unstructured Data Pipeline
Financial Fraud Detection System
"""

from enum import Enum
from typing import Dict, List, Set

# =============================================================================
# Entity Types
# =============================================================================

class EntityType(Enum):
    """Enumeration of all entity types for extraction"""
    # Corporate Structure
    COMPANY = "COMPANY"
    PARENT_COMPANY = "PARENT_COMPANY"
    SUBSIDIARY = "SUBSIDIARY"
    JOINT_VENTURE = "JOINT_VENTURE"
    RELATED_PARTY = "RELATED_PARTY"
    SHELL_COMPANY = "SHELL_COMPANY"
    
    # People
    PERSON = "PERSON"
    EXECUTIVE = "EXECUTIVE"
    BOARD_MEMBER = "BOARD_MEMBER"
    KEY_EMPLOYEE = "KEY_EMPLOYEE"
    FAMILY_RELATION = "FAMILY_RELATION"
    AUDITOR = "AUDITOR"
    
    # Financial
    FINANCIAL_TERM = "FINANCIAL_TERM"
    MONETARY_VALUE = "MONETARY_VALUE"
    REVENUE_SOURCE = "REVENUE_SOURCE"
    EXPENSE = "EXPENSE"
    ASSET = "ASSET"
    LIABILITY = "LIABILITY"
    ACCOUNTING_POLICY = "ACCOUNTING_POLICY"
    
    # Regulatory
    REGULATION = "REGULATION"
    SEC_FILING = "SEC_FILING"
    COMPLIANCE_ITEM = "COMPLIANCE_ITEM"
    
    # Temporal
    FISCAL_PERIOD = "FISCAL_PERIOD"
    TRANSACTION_DATE = "TRANSACTION_DATE"
    EVENT_DATE = "EVENT_DATE"


class RelationshipType(Enum):
    """Enumeration of relationship types between entities"""
    # Ownership & Control
    OWNS = "OWNS"
    CONTROLS = "CONTROLS"
    SUBSIDIARY_OF = "SUBSIDIARY_OF"
    PARENT_OF = "PARENT_OF"
    
    # Employment & Governance
    EMPLOYS = "EMPLOYS"
    WORKS_FOR = "WORKS_FOR"
    BOARD_MEMBER_OF = "BOARD_MEMBER_OF"
    OVERSEES = "OVERSEES"
    REPORTS_TO = "REPORTS_TO"
    
    # Transactions
    HAS_TRANSACTION_WITH = "HAS_TRANSACTION_WITH"
    PAYS = "PAYS"
    RECEIVES_FROM = "RECEIVES_FROM"
    OWES = "OWES"
    LOANS_TO = "LOANS_TO"
    
    # Disclosure
    DISCLOSED_IN = "DISCLOSED_IN"
    NOT_DISCLOSED_IN = "NOT_DISCLOSED_IN"
    OMITTED_FROM = "OMITTED_FROM"
    REFERENCES = "REFERENCES"
    
    # Compliance
    VIOLATES = "VIOLATES"
    COMPLIES_WITH = "COMPLIES_WITH"
    AUDITS = "AUDITS"
    QUALIFIES = "QUALIFIES"
    
    # Document Structure
    MENTIONED_IN = "MENTIONED_IN"
    CONTRADICTS = "CONTRADICTS"
    SUPPORTS = "SUPPORTS"
    
    # Family/Personal
    RELATED_TO = "RELATED_TO"
    FAMILY_OF = "FAMILY_OF"
    SPOUSE_OF = "SPOUSE_OF"
    
    # Financial
    COLLATERAL_FOR = "COLLATERAL_FOR"
    GUARANTEES = "GUARANTEES"
    SIGNS = "SIGNS"
    AUTHORS = "AUTHORS"


# =============================================================================
# SEC 10-K Section Mappings
# =============================================================================

SEC_10K_SECTIONS: Dict[str, Dict] = {
    "ITEM_1": {
        "title": "Business",
        "keywords": ["ITEM 1", "BUSINESS", "DESCRIPTION OF BUSINESS"],
        "entity_focus": ["subsidiaries", "segments", "products"],
        "fraud_relevance": "medium"
    },
    "ITEM_1A": {
        "title": "Risk Factors",
        "keywords": ["ITEM 1A", "RISK FACTORS"],
        "entity_focus": ["risks", "mitigations", "uncertainties"],
        "fraud_relevance": "high"
    },
    "ITEM_1B": {
        "title": "Unresolved Staff Comments",
        "keywords": ["ITEM 1B", "UNRESOLVED STAFF COMMENTS"],
        "entity_focus": ["sec_comments", "issues"],
        "fraud_relevance": "high"
    },
    "ITEM_2": {
        "title": "Properties",
        "keywords": ["ITEM 2", "PROPERTIES"],
        "entity_focus": ["real_estate", "facilities"],
        "fraud_relevance": "low"
    },
    "ITEM_3": {
        "title": "Legal Proceedings",
        "keywords": ["ITEM 3", "LEGAL PROCEEDINGS"],
        "entity_focus": ["lawsuits", "regulatory_actions"],
        "fraud_relevance": "high"
    },
    "ITEM_5": {
        "title": "Market for Common Equity",
        "keywords": ["ITEM 5", "MARKET FOR"],
        "entity_focus": ["stock", "dividends"],
        "fraud_relevance": "medium"
    },
    "ITEM_6": {
        "title": "Selected Financial Data",
        "keywords": ["ITEM 6", "SELECTED FINANCIAL DATA"],
        "entity_focus": ["financial_metrics", "trends"],
        "fraud_relevance": "high"
    },
    "ITEM_7": {
        "title": "Management's Discussion and Analysis (MD&A)",
        "keywords": ["ITEM 7", "MANAGEMENT'S DISCUSSION", "MD&A"],
        "entity_focus": ["performance", "explanations", "outlook"],
        "fraud_relevance": "critical"
    },
    "ITEM_7A": {
        "title": "Quantitative and Qualitative Disclosures About Market Risk",
        "keywords": ["ITEM 7A", "MARKET RISK"],
        "entity_focus": ["risk_exposure", "hedging"],
        "fraud_relevance": "medium"
    },
    "ITEM_8": {
        "title": "Financial Statements and Supplementary Data",
        "keywords": ["ITEM 8", "FINANCIAL STATEMENTS"],
        "entity_focus": ["numbers", "policies", "footnotes"],
        "fraud_relevance": "critical"
    },
    "ITEM_9": {
        "title": "Changes in and Disagreements with Accountants",
        "keywords": ["ITEM 9", "DISAGREEMENTS WITH ACCOUNTANTS"],
        "entity_focus": ["auditor_changes", "disputes"],
        "fraud_relevance": "critical"
    },
    "ITEM_9A": {
        "title": "Controls and Procedures",
        "keywords": ["ITEM 9A", "CONTROLS AND PROCEDURES"],
        "entity_focus": ["internal_controls", "weaknesses"],
        "fraud_relevance": "critical"
    },
    "FOOTNOTES": {
        "title": "Notes to Financial Statements",
        "keywords": ["NOTES TO", "FOOTNOTES", "NOTE 1", "NOTE 2"],
        "entity_focus": ["details", "qualifications", "policies"],
        "fraud_relevance": "critical"
    }
}


# =============================================================================
# Fraud Pattern Definitions
# =============================================================================

FRAUD_PATTERNS: Dict[str, Dict] = {
    "UNDISCLOSED_RELATIONSHIP": {
        "name": "Undisclosed Related Party Transaction",
        "description": "Hidden relationships between executives and business partners",
        "entities_required": ["EXECUTIVE", "RELATED_PARTY", "SUBSIDIARY"],
        "relationships_required": ["RELATED_TO", "HAS_TRANSACTION_WITH"],
        "detection_logic": "executive_mentioned AND supplier_mentioned AND relationship_not_disclosed",
        "risk_level": "HIGH",
        "sec_rule": "S-X 4-08(k)"
    },
    "SUBSIDIARY_HIDING": {
        "name": "Off-Balance Sheet Subsidiary",
        "description": "Subsidiary mentioned but not properly consolidated",
        "entities_required": ["SUBSIDIARY", "PARENT_COMPANY"],
        "relationships_required": ["SUBSIDIARY_OF", "DISCLOSED_IN"],
        "detection_logic": "subsidiary_in_text AND NOT_IN consolidation_footnote",
        "risk_level": "HIGH",
        "sec_rule": "S-X 4-08(g)"
    },
    "CONTRADICTION_DETECTION": {
        "name": "MD&A vs Footnote Contradiction",
        "description": "Statements in MD&A contradict footnote disclosures",
        "entities_required": ["FINANCIAL_TERM", "MONETARY_VALUE"],
        "relationships_required": ["CONTRADICTS", "MENTIONED_IN"],
        "detection_logic": "mda_statement AND footnote_statement AND semantic_contradiction",
        "risk_level": "HIGH",
        "sec_rule": "Reg S-K Item 303"
    },
    "REVENUE_MANIPULATION": {
        "name": "Revenue Recognition Irregularity",
        "description": "Unusual revenue recognition patterns or policy changes",
        "entities_required": ["REVENUE_SOURCE", "ACCOUNTING_POLICY", "FISCAL_PERIOD"],
        "relationships_required": ["REFERENCES", "MENTIONED_IN"],
        "detection_logic": "revenue_policy_change AND timing_unusual",
        "risk_level": "HIGH",
        "sec_rule": "ASC 606"
    },
    "CIRCULAR_REFERENCE": {
        "name": "Circular Transaction Pattern",
        "description": "Money flowing in circles between related entities",
        "entities_required": ["COMPANY", "SUBSIDIARY", "RELATED_PARTY"],
        "relationships_required": ["PAYS", "RECEIVES_FROM", "LOANS_TO"],
        "detection_logic": "transaction_chain_returns_to_origin",
        "risk_level": "CRITICAL",
        "sec_rule": "Multiple"
    },
    "EXECUTIVE_SELF_DEALING": {
        "name": "Executive Self-Dealing",
        "description": "Executives transacting with entities they control",
        "entities_required": ["EXECUTIVE", "COMPANY", "FAMILY_RELATION"],
        "relationships_required": ["CONTROLS", "HAS_TRANSACTION_WITH", "FAMILY_OF"],
        "detection_logic": "executive_controls_entity AND entity_has_transaction_with_company",
        "risk_level": "CRITICAL",
        "sec_rule": "S-K Item 404"
    },
    "AUDITOR_CONCERNS": {
        "name": "Auditor Red Flags",
        "description": "Qualified opinions, going concern, or auditor changes",
        "entities_required": ["AUDITOR", "COMPANY"],
        "relationships_required": ["AUDITS", "QUALIFIES"],
        "detection_logic": "qualified_opinion OR going_concern OR auditor_change",
        "risk_level": "CRITICAL",
        "sec_rule": "AU-C 570"
    },
    "SHELL_COMPANY_PATTERN": {
        "name": "Shell Company Structure",
        "description": "Entities with minimal operations used for transactions",
        "entities_required": ["SHELL_COMPANY", "COMPANY", "MONETARY_VALUE"],
        "relationships_required": ["HAS_TRANSACTION_WITH", "OWNS"],
        "detection_logic": "entity_minimal_disclosure AND significant_transactions",
        "risk_level": "HIGH",
        "sec_rule": "Multiple"
    }
}


# =============================================================================
# Pre-defined Fraud Detection Queries
# =============================================================================

FRAUD_QUERIES: List[str] = [
    "Find undisclosed related party transactions",
    "Identify contradictions in revenue recognition descriptions",
    "Detect changes in risk factors from previous year",
    "Find unusual relationships between executives and suppliers",
    "Identify subsidiaries mentioned but not in consolidation",
    "Find circular transaction patterns between entities",
    "Detect sudden changes in accounting policies",
    "Identify executives with undisclosed business interests",
    "Find shell companies or special purpose entities",
    "Detect auditor changes or qualified opinions",
    "Identify hidden liabilities or contingencies",
    "Find inconsistencies between MD&A and financial statements"
]


# =============================================================================
# Disclosure Requirements
# =============================================================================

DISCLOSURE_REQUIREMENTS: Dict[str, List[str]] = {
    "SUBSIDIARY": [
        "ownership_percentage",
        "consolidation_status",
        "financial_results",
        "jurisdiction",
        "principal_activities"
    ],
    "RELATED_PARTY": [
        "relationship_nature",
        "transaction_amount",
        "terms_and_conditions",
        "outstanding_balances"
    ],
    "EXECUTIVE": [
        "compensation",
        "stock_holdings",
        "related_transactions",
        "roles_and_responsibilities"
    ],
    "AUDITOR": [
        "firm_name",
        "opinion_type",
        "fees_paid",
        "tenure"
    ],
    "ACCOUNTING_POLICY": [
        "policy_description",
        "changes_from_prior_year",
        "impact_on_financials"
    ]
}


# =============================================================================
# Risk Levels
# =============================================================================

class RiskLevel(Enum):
    """Risk level classifications"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


RISK_SCORES: Dict[str, float] = {
    "CRITICAL": 1.0,
    "HIGH": 0.8,
    "MEDIUM": 0.5,
    "LOW": 0.3,
    "INFO": 0.1
}


# =============================================================================
# Graph Schema
# =============================================================================

NODE_PROPERTIES: Dict[str, List[str]] = {
    "COMPANY": ["name", "ticker", "industry", "jurisdiction", "year_founded"],
    "PERSON": ["name", "title", "role", "tenure_start", "tenure_end"],
    "FINANCIAL": ["type", "amount", "currency", "period", "section"],
    "DOCUMENT": ["filename", "filing_date", "fiscal_year", "form_type"],
    "SECTION": ["name", "page_start", "page_end", "word_count"]
}


EDGE_PROPERTIES: List[str] = [
    "weight",
    "confidence",
    "source_section",
    "source_page",
    "extraction_method",
    "timestamp"
]


# =============================================================================
# File Extensions
# =============================================================================

SUPPORTED_FILE_EXTENSIONS: Set[str] = {".txt", ".pdf"}


# =============================================================================
# Regex Patterns for Entity Extraction
# =============================================================================

REGEX_PATTERNS: Dict[str, str] = {
    "MONETARY_VALUE": r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|M|B|K))?",
    "PERCENTAGE": r"\d+(?:\.\d+)?%",
    "DATE": r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}",
    "FISCAL_YEAR": r"(?:fiscal\s+)?(?:year\s+)?(?:ended?\s+)?(?:ending\s+)?(?:FY\s*)?\d{4}",
    "FISCAL_QUARTER": r"Q[1-4]\s+\d{4}|(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?\d{4}",
    "SEC_ITEM": r"ITEM\s+\d+[A-Z]?\.?\s*[A-Z\s]+",
    "NOTE_REFERENCE": r"(?:Note|NOTE)\s+\d+",
    "TICKER_SYMBOL": r"\b[A-Z]{1,5}\b(?=\s*(?:stock|shares|common))",
}


# =============================================================================
# API Configuration
# =============================================================================

GROQ_MODELS: Dict[str, Dict] = {
    "llama-3.3-70b-versatile": {
        "context_window": 128000,
        "output_tokens": 32768,
        "best_for": "complex_analysis"
    },
    "llama-3.1-8b-instant": {
        "context_window": 128000,
        "output_tokens": 8192,
        "best_for": "fast_extraction"
    },
    "mixtral-8x7b-32768": {
        "context_window": 32768,
        "output_tokens": 32768,
        "best_for": "balanced"
    }
}
