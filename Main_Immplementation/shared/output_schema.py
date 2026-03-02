"""
Standardized output schema for multi-agent fraud detection system
"""
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class PipelineMetadata:
    """Metadata for pipeline execution"""
    pipeline_type: str  # 'structured', 'unstructured', or 'combined'
    timestamp: str
    version: str = "1.0.0"
    execution_time_seconds: Optional[float] = None
    records_processed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskAssessment:
    """Risk assessment from a single pipeline"""
    overall_risk_score: float  # 0-100
    risk_level: str  # CRITICAL, HIGH, MEDIUM, LOW, MINIMAL
    component_scores: Dict[str, float]
    risk_factors: List[str]
    confidence: float = 1.0  # Confidence in this assessment (0-1)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class UnifiedOutput:
    """
    Standardized output format for multi-agent system consumption
    """
    # Identification
    record_id: str
    source_identifier: Optional[str] = None  # Original filename, doc_id, etc.
    
    # Risk Assessments
    structured_risk: Optional[RiskAssessment] = None
    unstructured_risk: Optional[RiskAssessment] = None
    combined_risk: Optional[RiskAssessment] = None
    
    # Predictions (for structured pipeline)
    fraud_prediction: Optional[int] = None  # 0 or 1
    fraud_probability: Optional[float] = None  # 0-1
    
    # Entities and Relationships (for unstructured pipeline)
    entities: Optional[Dict[str, List[Dict]]] = None
    relationships: Optional[List[Dict]] = None
    
    # Multi-agent routing
    recommended_agents: List[str] = None
    priority: str = "normal"  # critical, high, normal, low
    processing_hints: List[str] = None
    requires_investigation: bool = False
    
    # Metadata
    metadata: Optional[PipelineMetadata] = None
    true_label: Optional[int] = None  # Ground truth if available
    
    def __post_init__(self):
        if self.recommended_agents is None:
            self.recommended_agents = []
        if self.processing_hints is None:
            self.processing_hints = []
        if self.metadata is None:
            self.metadata = PipelineMetadata(
                pipeline_type="unknown",
                timestamp=datetime.now().isoformat()
            )
        # Initialize extra fields storage
        if not hasattr(self, '_extra_fields'):
            self._extra_fields = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, handling nested dataclasses and extra fields"""
        # Use asdict for dataclass fields
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                result[key] = value
        
        # Add any extra fields (like agent_analysis) - these are not in the dataclass schema
        # Check _extra_fields stored on the instance (not in __dict__ from asdict)
        if hasattr(self, '_extra_fields'):
            result.update(self._extra_fields)
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedOutput':
        """Create from dictionary"""
        # Convert nested dicts back to dataclasses
        if 'structured_risk' in data and data['structured_risk']:
            data['structured_risk'] = RiskAssessment(**data['structured_risk'])
        if 'unstructured_risk' in data and data['unstructured_risk']:
            data['unstructured_risk'] = RiskAssessment(**data['unstructured_risk'])
        if 'combined_risk' in data and data['combined_risk']:
            data['combined_risk'] = RiskAssessment(**data['combined_risk'])
        if 'metadata' in data and data['metadata']:
            data['metadata'] = PipelineMetadata(**data['metadata'])
        
        return cls(**data)


class BatchOutput:
    """Container for batch processing results"""
    
    def __init__(self, batch_name: str = "fraud_detection_batch"):
        self.batch_name = batch_name
        self.timestamp = datetime.now().isoformat()
        self.records: List[UnifiedOutput] = []
        self.summary_stats: Dict[str, Any] = {}
    
    def add_record(self, record: UnifiedOutput):
        """Add a record to the batch"""
        self.records.append(record)
    
    def calculate_summary_stats(self):
        """Calculate summary statistics for the batch"""
        if not self.records:
            return
        
        total = len(self.records)
        
        # Count by risk level (use combined if available, otherwise use available pipeline)
        risk_levels = {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'MINIMAL': 0}
        fraud_predictions = 0
        
        risk_scores = []
        
        for record in self.records:
            # Get the most relevant risk assessment
            risk = record.combined_risk or record.structured_risk or record.unstructured_risk
            
            if risk:
                risk_levels[risk.risk_level] = risk_levels.get(risk.risk_level, 0) + 1
                risk_scores.append(risk.overall_risk_score)
            
            if record.fraud_prediction == 1:
                fraud_predictions += 1
        
        self.summary_stats = {
            'total_records': total,
            'fraud_predictions': fraud_predictions,
            'fraud_percentage': (fraud_predictions / total * 100) if total > 0 else 0,
            'risk_level_distribution': risk_levels,
            'average_risk_score': sum(risk_scores) / len(risk_scores) if risk_scores else 0,
            'max_risk_score': max(risk_scores) if risk_scores else 0,
            'min_risk_score': min(risk_scores) if risk_scores else 0,
            'critical_count': risk_levels['CRITICAL'],
            'high_risk_count': risk_levels['HIGH'] + risk_levels['CRITICAL']
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch to dictionary"""
        self.calculate_summary_stats()
        
        return {
            'batch_metadata': {
                'batch_name': self.batch_name,
                'timestamp': self.timestamp,
                'total_records': len(self.records)
            },
            'summary_statistics': self.summary_stats,
            'records': [record.to_dict() for record in self.records]
        }
    
    def save(self, output_path: str):
        """Save batch to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        
        return output_path
