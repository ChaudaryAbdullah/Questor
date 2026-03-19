"""
FinBERT Wrapper Module
Provides financial-specific NLP capabilities using FinBERT and transformers.
"""

import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

from ..utils.logger import get_nlp_logger
from ..utils.config_manager import get_config


@dataclass
class SentimentResult:
    """Result of sentiment analysis."""
    label: str  # positive, negative, neutral
    score: float
    confidence: float


@dataclass
class FinancialEntity:
    """Extracted financial entity."""
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    context: str


class FinBERTWrapper:
    """
    Wrapper for FinBERT model providing financial NLP capabilities.
    Supports sentiment analysis and financial entity recognition.
    """
    
    # Financial sentiment labels
    SENTIMENT_LABELS = ["positive", "negative", "neutral"]
    
    # Financial terms for pattern matching
    FINANCIAL_PATTERNS = {
        "growth_positive": [
            "increased", "growth", "improved", "stronger", "exceeded",
            "outperformed", "record", "robust", "solid", "healthy"
        ],
        "growth_negative": [
            "decreased", "declined", "weakened", "lower", "missed",
            "underperformed", "challenging", "difficult", "weak"
        ],
        "risk_indicators": [
            "material weakness", "going concern", "restatement",
            "significant deficiency", "adverse opinion", "disclaimer",
            "qualified opinion", "uncertainty", "litigation", "investigation"
        ],
        "fraud_keywords": [
            "related party", "off-balance sheet", "special purpose",
            "undisclosed", "restated", "correction", "irregular",
            "unusual", "non-recurring", "one-time"
        ]
    }
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize FinBERT wrapper.
        
        Args:
            model_name: HuggingFace model name
            device: torch device (cuda/cpu)
            cache_dir: Directory to cache model
        """
        self.logger = get_nlp_logger()
        self.model_name = model_name
        self.cache_dir = cache_dir
        
        # Determine device
        if device:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        
        self._model = None
        self._tokenizer = None
        self._initialized = False
        
        self.logger.info(f"FinBERT wrapper initialized (device: {self.device})")
    
    def _load_model(self) -> None:
        """Load the FinBERT model lazily."""
        if self._initialized:
            return
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            self.logger.info(f"Loading FinBERT model: {self.model_name}")
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            ).to(self.device)
            
            self._model.eval()
            self._initialized = True
            
            self.logger.info("FinBERT model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load FinBERT: {e}")
            raise
    
    def analyze_sentiment(
        self,
        text: str,
        return_all_scores: bool = False
    ) -> Union[SentimentResult, Dict[str, float]]:
        """
        Analyze financial sentiment of text.
        
        Args:
            text: Text to analyze
            return_all_scores: Return scores for all labels
        
        Returns:
            SentimentResult or dict of all scores
        """
        self._load_model()
        
        try:
            # Tokenize
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self._model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                scores = scores.cpu().numpy()[0]
            
            if return_all_scores:
                return {
                    label: float(score)
                    for label, score in zip(self.SENTIMENT_LABELS, scores)
                }
            
            # Get best prediction
            best_idx = np.argmax(scores)
            
            return SentimentResult(
                label=self.SENTIMENT_LABELS[best_idx],
                score=float(scores[best_idx]),
                confidence=float(scores[best_idx])
            )
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return SentimentResult(label="neutral", score=0.5, confidence=0.0)
    
    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[SentimentResult]:
        """
        Analyze sentiment for batch of texts.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
        
        Returns:
            List of SentimentResult objects
        """
        self._load_model()
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    scores = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    scores = scores.cpu().numpy()
                
                for score_row in scores:
                    best_idx = np.argmax(score_row)
                    results.append(SentimentResult(
                        label=self.SENTIMENT_LABELS[best_idx],
                        score=float(score_row[best_idx]),
                        confidence=float(score_row[best_idx])
                    ))
                    
            except Exception as e:
                self.logger.error(f"Batch sentiment failed: {e}")
                # Add neutral results for failed batch
                for _ in batch:
                    results.append(SentimentResult(
                        label="neutral",
                        score=0.5,
                        confidence=0.0
                    ))
        
        return results
    
    def detect_risk_language(self, text: str) -> Dict[str, List[str]]:
        """
        Detect risk and fraud-related language patterns.
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary of found patterns by category
        """
        text_lower = text.lower()
        found_patterns = {}
        
        for category, patterns in self.FINANCIAL_PATTERNS.items():
            matches = []
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    matches.append(pattern)
            
            if matches:
                found_patterns[category] = matches
        
        return found_patterns
    
    def score_fraud_risk(self, text: str) -> Tuple[float, List[str]]:
        """
        Score text for fraud risk indicators.
        
        Args:
            text: Text to analyze
        
        Returns:
            Tuple of (risk_score, list of indicators found)
        """
        patterns = self.detect_risk_language(text)
        
        indicators = []
        score = 0.0
        
        # Weight different pattern categories
        weights = {
            "fraud_keywords": 0.3,
            "risk_indicators": 0.25,
            "growth_negative": 0.1
        }
        
        for category, matches in patterns.items():
            if category in weights:
                score += len(matches) * weights[category]
                indicators.extend([f"{category}: {m}" for m in matches])
        
        # Normalize score to 0-1
        score = min(1.0, score)
        
        return score, indicators
    
    def extract_financial_entities(
        self,
        text: str
    ) -> List[FinancialEntity]:
        """
        Extract financial entities using pattern matching.
        
        Args:
            text: Text to analyze
        
        Returns:
            List of FinancialEntity objects
        """
        import re
        
        entities = []
        
        # Monetary values
        money_pattern = r"\$[\d,]+(?:\.\d{2})?(?:\s*(?:million|billion|thousand|M|B|K))?"
        for match in re.finditer(money_pattern, text):
            entities.append(FinancialEntity(
                text=match.group(),
                entity_type="MONETARY_VALUE",
                start=match.start(),
                end=match.end(),
                confidence=0.9,
                context=text[max(0, match.start()-50):match.end()+50]
            ))
        
        # Percentages
        pct_pattern = r"\d+(?:\.\d+)?%"
        for match in re.finditer(pct_pattern, text):
            entities.append(FinancialEntity(
                text=match.group(),
                entity_type="PERCENTAGE",
                start=match.start(),
                end=match.end(),
                confidence=0.9,
                context=text[max(0, match.start()-50):match.end()+50]
            ))
        
        # Fiscal periods
        fiscal_pattern = r"(?:fiscal\s+)?(?:year\s+)?(?:ended?\s+)?(?:ending\s+)?(?:FY\s*)?\d{4}|Q[1-4]\s+\d{4}"
        for match in re.finditer(fiscal_pattern, text, re.IGNORECASE):
            entities.append(FinancialEntity(
                text=match.group(),
                entity_type="FISCAL_PERIOD",
                start=match.start(),
                end=match.end(),
                confidence=0.8,
                context=text[max(0, match.start()-50):match.end()+50]
            ))
        
        return entities
    
    def compare_sections(
        self,
        text1: str,
        text2: str,
        section1_name: str = "Section 1",
        section2_name: str = "Section 2"
    ) -> Dict:
        """
        Compare sentiment between two sections.
        
        Args:
            text1: First text section
            text2: Second text section
            section1_name: Name of first section
            section2_name: Name of second section
        
        Returns:
            Comparison results
        """
        sent1 = self.analyze_sentiment(text1, return_all_scores=True)
        sent2 = self.analyze_sentiment(text2, return_all_scores=True)
        
        risk1, ind1 = self.score_fraud_risk(text1)
        risk2, ind2 = self.score_fraud_risk(text2)
        
        # Calculate difference
        sentiment_diff = {
            label: sent1[label] - sent2[label]
            for label in self.SENTIMENT_LABELS
        }
        
        return {
            section1_name: {
                "sentiment": sent1,
                "fraud_risk_score": risk1,
                "indicators": ind1
            },
            section2_name: {
                "sentiment": sent2,
                "fraud_risk_score": risk2,
                "indicators": ind2
            },
            "difference": {
                "sentiment_diff": sentiment_diff,
                "risk_diff": risk1 - risk2
            },
            "contradiction_likely": abs(sentiment_diff["positive"] - sentiment_diff["negative"]) > 0.3
        }
    
    def is_available(self) -> bool:
        """Check if FinBERT is available and loadable."""
        try:
            from transformers import AutoModelForSequenceClassification
            return True
        except ImportError:
            return False
