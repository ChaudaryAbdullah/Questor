"""
Centralized Logging Module for the Unstructured Data Pipeline
Provides structured logging with file and console handlers.
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


class PipelineLogger:
    """
    Centralized logger for the fraud detection pipeline.
    Supports both file and console logging with configurable levels.
    """
    
    _instances: dict = {}
    _initialized: bool = False
    _log_dir: Path = Path("logs")
    
    def __new__(cls, name: str = "pipeline", **kwargs):
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]
    
    def __init__(
        self,
        name: str = "pipeline",
        level: str = "INFO",
        log_dir: Optional[str] = None,
        max_bytes: int = 10485760,  # 10 MB
        backup_count: int = 5,
        console_output: bool = True,
        file_output: bool = True
    ):
        """
        Initialize the pipeline logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_dir: Directory for log files
            max_bytes: Maximum size of each log file
            backup_count: Number of backup files to keep
            console_output: Enable console logging
            file_output: Enable file logging
        """
        if hasattr(self, '_logger') and self._logger:
            return
            
        self.name = name
        self._logger = logging.getLogger(name)
        self._logger.setLevel(getattr(logging, level.upper()))
        self._logger.handlers = []  # Clear existing handlers
        
        # Log format
        self.format_string = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
        self.formatter = logging.Formatter(self.format_string)
        
        # Console handler
        if console_output:
            self._add_console_handler()
        
        # File handler
        if file_output:
            if log_dir:
                self._log_dir = Path(log_dir)
            self._add_file_handler(max_bytes, backup_count)
    
    def _add_console_handler(self) -> None:
        """Add console handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.formatter)
        self._logger.addHandler(console_handler)
    
    def _add_file_handler(
        self,
        max_bytes: int,
        backup_count: int
    ) -> None:
        """Add rotating file handler."""
        self._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d")
        log_file = self._log_dir / f"{self.name}_{timestamp}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(self.formatter)
        self._logger.addHandler(file_handler)
    
    @property
    def logger(self) -> logging.Logger:
        """Get the underlying logger instance."""
        return self._logger
    
    def debug(self, message: str, *args, **kwargs) -> None:
        """Log debug message."""
        self._logger.debug(message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs) -> None:
        """Log info message."""
        self._logger.info(message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs) -> None:
        """Log warning message."""
        self._logger.warning(message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs) -> None:
        """Log error message."""
        self._logger.error(message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs) -> None:
        """Log critical message."""
        self._logger.critical(message, *args, **kwargs)
    
    def exception(self, message: str, *args, **kwargs) -> None:
        """Log exception with traceback."""
        self._logger.exception(message, *args, **kwargs)
    
    def set_level(self, level: str) -> None:
        """Change log level dynamically."""
        self._logger.setLevel(getattr(logging, level.upper()))
    
    def log_pipeline_start(self, document_path: str) -> None:
        """Log pipeline start with document info."""
        self.info("=" * 60)
        self.info(f"PIPELINE START - Processing: {document_path}")
        self.info("=" * 60)
    
    def log_pipeline_end(self, document_path: str, success: bool, duration: float) -> None:
        """Log pipeline completion."""
        status = "SUCCESS" if success else "FAILED"
        self.info("=" * 60)
        self.info(f"PIPELINE {status} - {document_path}")
        self.info(f"Total Duration: {duration:.2f} seconds")
        self.info("=" * 60)
    
    def log_module_start(self, module_name: str) -> None:
        """Log module processing start."""
        self.info(f"[{module_name}] Starting...")
    
    def log_module_end(self, module_name: str, duration: float) -> None:
        """Log module processing completion."""
        self.info(f"[{module_name}] Completed in {duration:.2f}s")
    
    def log_entity_extraction(
        self,
        entity_type: str,
        count: int,
        chunk_id: Optional[str] = None
    ) -> None:
        """Log entity extraction results."""
        if chunk_id:
            self.debug(f"Extracted {count} {entity_type} entities from chunk {chunk_id}")
        else:
            self.info(f"Extracted {count} {entity_type} entities total")
    
    def log_fraud_indicator(
        self,
        indicator_type: str,
        risk_level: str,
        details: str
    ) -> None:
        """Log detected fraud indicator."""
        self.warning(f"FRAUD INDICATOR [{risk_level}] - {indicator_type}: {details}")
    
    def log_api_call(
        self,
        provider: str,
        endpoint: str,
        success: bool,
        duration: float
    ) -> None:
        """Log API call details."""
        status = "OK" if success else "FAILED"
        self.debug(f"API Call [{provider}] {endpoint} - {status} ({duration:.2f}s)")


def get_logger(
    name: str = "pipeline",
    level: str = "INFO",
    **kwargs
) -> PipelineLogger:
    """
    Get or create a pipeline logger instance.
    
    Args:
        name: Logger name
        level: Logging level
        **kwargs: Additional arguments for PipelineLogger
    
    Returns:
        PipelineLogger instance
    """
    return PipelineLogger(name=name, level=level, **kwargs)


# Create default logger instance
default_logger = get_logger("fraud_detection_pipeline")


# Module-specific loggers
def get_preprocessing_logger() -> PipelineLogger:
    """Get logger for preprocessing module."""
    return get_logger("preprocessing")


def get_nlp_logger() -> PipelineLogger:
    """Get logger for NLP module."""
    return get_logger("nlp")


def get_graph_logger() -> PipelineLogger:
    """Get logger for graph module."""
    return get_logger("graph")


def get_retrieval_logger() -> PipelineLogger:
    """Get logger for retrieval module."""
    return get_logger("retrieval")


def get_validation_logger() -> PipelineLogger:
    """Get logger for validation module."""
    return get_logger("validation")
