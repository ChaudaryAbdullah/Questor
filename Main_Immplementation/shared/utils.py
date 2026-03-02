"""
Shared utility functions for pipeline infrastructure
"""
import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional


def resolve_path(path: str, base_dir: Optional[str] = None) -> Path:
    """
    Resolve a path relative to a base directory
    
    Args:
        path: Path to resolve
        base_dir: Base directory (defaults to Pipelines directory)
    
    Returns:
        Resolved Path object
    """
    if base_dir is None:
        # Default to Pipelines directory
        base_dir = Path(__file__).parent.parent
    else:
        base_dir = Path(base_dir)
    
    path_obj = Path(path)
    
    if path_obj.is_absolute():
        return path_obj
    else:
        return (base_dir / path_obj).resolve()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file (defaults to config.yaml in Pipelines dir)
    
    Returns:
        Configuration dictionary with defaults
    """
    # Default configuration
    default_config = {
        'pipelines': {
            'structured': {
                'enabled': True,
                'entry_point': 'stuctured_pipeline/run_inference.py',
                'weight': 0.6
            },
            'unstructured': {
                'enabled': True,
                'entry_point': 'unstructured_pipeline/main.py',
                'weight': 0.4
            }
        },
        'output': {
            'base_dir': 'output',
            'structured_dir': 'output/structured',
            'unstructured_dir': 'output/unstructured',
            'combined_dir': 'output/combined',
            'multiagent_dir': 'output/multiagent_ready'
        },
        'score_combination': {
            'method': 'weighted_average',
            'structured_weight': 0.6,
            'unstructured_weight': 0.4,
            'conflict_threshold': 30,  # Flag if scores differ by more than this
            'missing_penalty': 0.8  # Multiply single-source score by this if other is missing
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    }
    
    # Try to load from file
    if config_path is None:
        config_path = resolve_path('config.yaml')
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
            
            # Merge with defaults (user config overrides defaults)
            config = deep_update(default_config, user_config)
            return config
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
            print("Using default configuration")
    
    return default_config


def deep_update(base_dict: Dict, update_dict: Dict) -> Dict:
    """
    Deep merge two dictionaries
    
    Args:
        base_dict: Base dictionary
        update_dict: Dictionary with updates
    
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    
    return result


def setup_logging(level: str = 'INFO', log_format: Optional[str] = None) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log message format
    
    Returns:
        Configured logger
    """
    if log_format is None:
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger('UnifiedPipeline')
    return logger


def ensure_directory(path: str) -> Path:
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time string
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"
