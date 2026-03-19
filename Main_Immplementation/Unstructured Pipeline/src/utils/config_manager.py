"""
Configuration Manager for the Unstructured Data Pipeline
Handles loading, validation, and access to pipeline configuration.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml

from .logger import get_logger


class ConfigManager:
    """
    Centralized configuration manager for the fraud detection pipeline.
    Loads configuration from YAML files and environment variables.
    """
    
    _instance: Optional['ConfigManager'] = None
    _config: Dict[str, Any] = {}
    
    def __new__(cls, config_path: Optional[str] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the settings.yaml file
        """
        if self._initialized:
            return
            
        self.logger = get_logger("config_manager")
        self._config_path = config_path or self._find_config_file()
        self._load_config()
        self._initialized = True
    
    def _find_config_file(self) -> Path:
        """Find the configuration file in standard locations."""
        # Try multiple locations
        possible_paths = [
            Path("config/settings.yaml"),
            Path("../config/settings.yaml"),
            Path(__file__).parent.parent.parent / "config" / "settings.yaml",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(
            "Configuration file not found. Please ensure config/settings.yaml exists."
        )
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {self._config_path}")
            self._resolve_env_variables()
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _resolve_env_variables(self) -> None:
        """Resolve environment variables in configuration."""
        self._resolve_env_recursive(self._config)
    
    def _resolve_env_recursive(self, config: Union[Dict, list, str]) -> None:
        """Recursively resolve environment variables."""
        if isinstance(config, dict):
            # Collect changes to apply after iteration
            updates = {}
            for key, value in config.items():
                if isinstance(value, str) and key.endswith("_env"):
                    # This is an environment variable reference
                    env_value = os.getenv(value)
                    if env_value:
                        # Store resolved value under key without _env suffix
                        new_key = key.replace("_env", "")
                        updates[new_key] = env_value
                elif isinstance(value, (dict, list)):
                    self._resolve_env_recursive(value)
            # Apply updates after iteration
            config.update(updates)
        elif isinstance(config, list):
            for item in config:
                self._resolve_env_recursive(item)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key: Configuration key (e.g., "llm.provider")
            default: Default value if key not found
        
        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get entire configuration section.
        
        Args:
            section: Section name (e.g., "llm", "graph_db")
        
        Returns:
            Dictionary with section configuration
        """
        return self._config.get(section, {})
    
    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration."""
        config = self.get_section("llm")
        # Ensure API key is resolved
        if "api_key" not in config and "api_key_env" in config:
            config["api_key"] = os.getenv(config["api_key_env"], "")
        return config
    
    @property
    def groq_api_key(self) -> str:
        """Get Groq API key."""
        return os.getenv(
            self.get("llm.api_key_env", "GROQ_API_KEY"),
            ""
        )
    
    @property
    def embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return self.get_section("embeddings")
    
    @property
    def vector_db_config(self) -> Dict[str, Any]:
        """Get vector database configuration."""
        return self.get_section("vector_db")
    
    @property
    def graph_db_config(self) -> Dict[str, Any]:
        """Get graph database configuration."""
        config = self.get_section("graph_db")
        # Resolve Neo4j password
        if "password" not in config and "password_env" in config:
            config["password"] = os.getenv(config["password_env"], "")
        return config
    
    @property
    def neo4j_uri(self) -> str:
        """Get Neo4j connection URI."""
        return self.get("graph_db.uri", "bolt://localhost:7687")
    
    @property
    def neo4j_credentials(self) -> tuple:
        """Get Neo4j username and password."""
        username = self.get("graph_db.username", "neo4j")
        password = os.getenv(
            self.get("graph_db.password_env", "NEO4J_PASSWORD"),
            ""
        )
        return username, password
    
    @property
    def preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.get_section("preprocessing")
    
    @property
    def entity_extraction_config(self) -> Dict[str, Any]:
        """Get entity extraction configuration."""
        return self.get_section("entity_extraction")
    
    @property
    def rag_config(self) -> Dict[str, Any]:
        """Get RAG configuration."""
        return self.get_section("rag")
    
    @property
    def fraud_detection_config(self) -> Dict[str, Any]:
        """Get fraud detection configuration."""
        return self.get_section("fraud_detection")
    
    @property
    def visualization_config(self) -> Dict[str, Any]:
        """Get visualization configuration."""
        return self.get_section("visualization")
    
    @property
    def data_dirs(self) -> Dict[str, Path]:
        """Get data directory paths."""
        data_config = self.get_section("data")
        base_path = Path(__file__).parent.parent.parent
        
        return {
            key: base_path / path
            for key, path in data_config.items()
        }
    
    def ensure_directories(self) -> None:
        """Create all required data directories."""
        for name, path in self.data_dirs.items():
            path.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {path}")
    
    def validate_config(self) -> bool:
        """
        Validate that all required configuration values are present.
        
        Returns:
            True if configuration is valid
        """
        required_sections = ["llm", "vector_db", "graph_db", "preprocessing"]
        
        for section in required_sections:
            if section not in self._config:
                self.logger.error(f"Missing required configuration section: {section}")
                return False
        
        # Validate API key is available
        if not self.groq_api_key:
            self.logger.warning(
                "Groq API key not found. Set GROQ_API_KEY environment variable."
            )
        
        # Validate Neo4j credentials
        username, password = self.neo4j_credentials
        if not password:
            self.logger.warning(
                "Neo4j password not found. Set NEO4J_PASSWORD environment variable."
            )
        
        return True
    
    def reload(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        self.logger.info("Configuration reloaded")
    
    def to_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary."""
        return self._config.copy()
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the configuration manager instance.
    
    Args:
        config_path: Optional path to configuration file
    
    Returns:
        ConfigManager instance
    """
    return ConfigManager(config_path)
