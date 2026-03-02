"""
Shared utilities and infrastructure for fraud detection pipelines
"""

__version__ = "1.0.0"

from .output_schema import UnifiedOutput, RiskAssessment, PipelineMetadata
from .utils import load_config, setup_logging, resolve_path

__all__ = [
    'UnifiedOutput',
    'RiskAssessment',
    'PipelineMetadata',
    'load_config',
    'setup_logging',
    'resolve_path'
]
