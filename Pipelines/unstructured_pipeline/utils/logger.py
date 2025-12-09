
# File: utils/logger.py
"""
Logging utility for the system
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
from .config import Config

class Logger:
    """Custom logger for the fraud detection system"""
    
    _loggers = {}
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get or create a logger instance"""
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        
        # File handler
        log_file = Config.LOGS_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_format)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger


