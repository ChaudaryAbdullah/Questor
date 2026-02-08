"""
Agent Orchestrator
Coordinates execution of multiple fraud detection agents
"""
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import importlib
import yaml

from .base_agent import BaseAgent, AgentResult


class AgentOrchestrator:
    """
    Orchestrates execution of multiple fraud detection agents
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize orchestrator
        
        Args:
            config: Configuration dictionary with agent settings
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.agents: Dict[str, BaseAgent] = {}
        self._register_agents()
    
    def _register_agents(self):
        """Discover and register available agents"""
        # Import and register built-in agents
        try:
            from .benfords_law import BenfordsLawAgent
            agent_config = self.config.get('agents', {}).get('benfords_law', {})
            self.agents['benfords_law'] = BenfordsLawAgent(agent_config)
            self.logger.info("Registered Benford's Law agent")
        except Exception as e:
            self.logger.warning(f"Failed to register Benford's Law agent: {e}")
        
        try:
            from .beneish_mscore import BeneishMScoreAgent
            agent_config = self.config.get('agents', {}).get('beneish_mscore', {})
            self.agents['beneish_mscore'] = BeneishMScoreAgent(agent_config)
            self.logger.info("Registered Beneish M-Score agent")
        except Exception as e:
            self.logger.warning(f"Failed to register Beneish M-Score agent: {e}")
    
    def run_agents(
        self,
        data: Dict[str, Any],
        enabled_agents: Optional[List[str]] = None,
        disabled_agents: Optional[List[str]] = None
    ) -> Dict[str, AgentResult]:
        """
        Run all enabled agents on the data
        
        Args:
            data: Input data to analyze
            enabled_agents: List of agent names to enable (None = all)
            disabled_agents: List of agent names to disable
        
        Returns:
            Dictionary of agent_name -> AgentResult
        """
        results = {}
        disabled_agents = disabled_agents or []
        
        for name, agent in self.agents.items():
            # Check if agent should run
            if enabled_agents is not None and name not in enabled_agents:
                self.logger.debug(f"Skipping {name} - not in enabled list")
                continue
            
            if name in disabled_agents:
                self.logger.debug(f"Skipping {name} - explicitly disabled")
                continue
            
            if not agent.is_available():
                self.logger.debug(f"Skipping {name} - not available")
                continue
            
            # Run agent
            self.logger.info(f"Running agent: {name}")
            result = agent.safe_analyze(data)
            results[name] = result
            
            if result.success:
                self.logger.info(f"  ✓ {name}: score={result.score:.2f}, confidence={result.confidence:.2f}")
            else:
                self.logger.warning(f"  ✗ {name}: {result.error}")
        
        return results
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agent names"""
        return [name for name, agent in self.agents.items() if agent.is_available()]
    
    def get_all_agents(self) -> List[str]:
        """Get list of all registered agent names"""
        return list(self.agents.keys())
    
    def get_agent_weights(self, results: Dict[str, AgentResult]) -> Dict[str, float]:
        """
        Get weights for successful agents
        
        Args:
            results: Agent results
        
        Returns:
            Dictionary of agent_name -> weight
        """
        weights = {}
        
        for name, result in results.items():
            if result.success and name in self.agents:
                weights[name] = self.agents[name].get_weight()
        
        return weights
    
    def calculate_combined_score(
        self,
        results: Dict[str, AgentResult],
        normalize_weights: bool = True
    ) -> Dict[str, Any]:
        """
        Calculate weighted combined score from agent results
        
        Args:
            results: Agent results
            normalize_weights: Whether to normalize weights to sum to 1.0
        
        Returns:
            Dictionary with combined score and details
        """
        # Get weights for successful agents
        weights = self.get_agent_weights(results)
        
        if not weights:
            return {
                'combined_score': 0.0,
                'confidence': 0.0,
                'weights_used': {},
                'agents_succeeded': 0,
                'agents_failed': len(results)
            }
        
        # Normalize weights if requested
        if normalize_weights:
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {name: w / total_weight for name, w in weights.items()}
        
        # Calculate weighted score
        combined_score = 0.0
        combined_confidence = 0.0
        
        for name, result in results.items():
            if result.success and name in weights:
                weight = weights[name]
                combined_score += result.score * weight
                combined_confidence += result.confidence * weight
        
        return {
            'combined_score': round(combined_score, 2),
            'confidence': round(combined_confidence, 2),
            'weights_used': {k: round(v, 3) for k, v in weights.items()},
            'agents_succeeded': sum(1 for r in results.values() if r.success),
            'agents_failed': sum(1 for r in results.values() if not r.success)
        }


def load_agent_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load agent configuration from YAML file
    
    Args:
        config_path: Path to config file (uses default if None)
    
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / 'agent_config.yaml'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logging.warning(f"Failed to load agent config: {e}")
        return {}
