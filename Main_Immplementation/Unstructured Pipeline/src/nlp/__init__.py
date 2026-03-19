"""
NLP module for entity extraction and embeddings.
"""

from .entity_extractor import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedRelationship,
    ExtractionResult
)

from .finbert_wrapper import (
    FinBERTWrapper,
    SentimentResult,
    FinancialEntity as FinBERTFinancialEntity
)

__all__ = [
    "EntityExtractor",
    "ExtractedEntity",
    "ExtractedRelationship",
    "ExtractionResult",
    "FinBERTWrapper",
    "SentimentResult"
]
