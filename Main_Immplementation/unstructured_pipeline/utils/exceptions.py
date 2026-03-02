# File: utils/exceptions.py
"""
Custom exceptions for the fraud detection system
"""

class FraudDetectionException(Exception):
    """Base exception for fraud detection system"""
    pass

class DataIngestionError(FraudDetectionException):
    """Raised when data ingestion fails"""
    pass

class EmbeddingError(FraudDetectionException):
    """Raised when embedding generation fails"""
    pass

class VectorDBError(FraudDetectionException):
    """Raised when vector database operations fail"""
    pass

class GraphDBError(FraudDetectionException):
    """Raised when graph database operations fail"""
    pass

class NERExtractionError(FraudDetectionException):
    """Raised when NER extraction fails"""
    pass


