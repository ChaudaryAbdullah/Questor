"""
Base Agent Module
Defines the abstract base class for all fraud detection agents
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging


@dataclass
class AgentResult:
    """
    Standardized result format for all agents
    """
    agent_name: str
    score: float  # Risk score 0-100
    confidence: float  # Confidence 0-1
    findings: List[str]  # List of findings/anomalies
    metrics: Dict[str, Any]  # Agent-specific metrics
    timestamp: str
    success: bool = True
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class BaseAgent(ABC):
    """
    Abstract base class for all fraud detection agents
    
    Each agent must implement:
    - analyze(): Main analysis method
    - get_name(): Return agent name
    - get_weight(): Return default weight for score combination
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize agent
        
        Args:
            config: Agent-specific configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.enabled = self.config.get('enabled', True)
        self._weight = self.config.get('weight', 0.1)
    
    @abstractmethod
    def analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Analyze data and return risk assessment
        
        Args:
            data: Input data dictionary (from structured pipeline)
        
        Returns:
            AgentResult with score and findings
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return agent name"""
        pass
    
    def get_weight(self) -> float:
        """Return default weight for score combination"""
        return self._weight
    
    def is_available(self) -> bool:
        """Check if agent is available and enabled"""
        return self.enabled
    
    def is_applicable(self, data: Dict[str, Any]) -> bool:
        """
        Check if agent can analyze the given data
        
        Args:
            data: Input data
        
        Returns:
            True if agent can process this data
        """
        return True
    
    def _create_result(
        self,
        score: float,
        confidence: float,
        findings: List[str],
        metrics: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None
    ) -> AgentResult:
        """Helper to create standardized result"""
        return AgentResult(
            agent_name=self.get_name(),
            score=max(0.0, min(100.0, score)),  # Clamp to 0-100
            confidence=max(0.0, min(1.0, confidence)),  # Clamp to 0-1
            findings=findings,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            success=success,
            error=error
        )
    
    def safe_analyze(self, data: Dict[str, Any]) -> AgentResult:
        """
        Safely execute analysis with error handling
        
        Args:
            data: Input data
        
        Returns:
            AgentResult (with error info if failed)
        """
        try:
            if not self.is_available():
                return self._create_result(
                    score=0.0,
                    confidence=0.0,
                    findings=[],
                    metrics={},
                    success=False,
                    error="Agent is disabled"
                )
            
            if not self.is_applicable(data):
                return self._create_result(
                    score=0.0,
                    confidence=0.0,
                    findings=["Data not applicable for this agent"],
                    metrics={},
                    success=False,
                    error="Data not applicable"
                )
            
            return self.analyze(data)
            
        except Exception as e:
            self.logger.error(f"Agent {self.get_name()} failed: {e}", exc_info=True)
            return self._create_result(
                score=0.0,
                confidence=0.0,
                findings=[],
                metrics={},
                success=False,
                error=str(e)
            )
