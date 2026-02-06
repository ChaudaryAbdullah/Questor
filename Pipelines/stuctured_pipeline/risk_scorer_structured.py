# File: Pipelines/stuctured_pipeline/risk_scorer_structured.py
"""
Risk Scorer for Structured Pipeline
Calculates fraud risk scores based on model predictions and confidence levels
"""
from typing import Dict, List, Any, Tuple
import numpy as np
import pandas as pd
from datetime import datetime


class StructuredRiskScorer:
    """
    Calculate risk scores for structured data based on ensemble predictions
    """
    
    # Risk component weights
    WEIGHTS = {
        'model_confidence': 0.40,      # Ensemble confidence
        'model_agreement': 0.30,        # Agreement among models
        'prediction_strength': 0.20,    # How strongly models predict fraud
        'anomaly_indicators': 0.10      # Statistical anomalies in features
    }
    
    def __init__(self):
        """Initialize risk scorer"""
        self.logger_enabled = True
    
    def log(self, message: str):
        """Simple logging"""
        if self.logger_enabled:
            print(f"[RiskScorer] {message}")
    
    def calculate_risk_score(
        self,
        ensemble_prob: float,
        individual_probs: Dict[str, float],
        individual_preds: Dict[str, int],
        feature_values: Dict[str, float] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk score for a single record
        
        Args:
            ensemble_prob: Ensemble fraud probability (0-1)
            individual_probs: Dict of model_name -> probability
            individual_preds: Dict of model_name -> prediction (0/1)
            feature_values: Optional dict of feature values for anomaly detection
            
        Returns:
            Dictionary with risk assessment
        """
        # Component 1: Model Confidence (0-100)
        confidence_score = ensemble_prob * 100
        
        # Component 2: Model Agreement (0-100)
        if individual_preds:
            fraud_votes = sum(individual_preds.values())
            total_models = len(individual_preds)
            agreement_ratio = fraud_votes / total_models if total_models > 0 else 0
            agreement_score = agreement_ratio * 100
        else:
            agreement_score = 50.0
        
        # Component 3: Prediction Strength (0-100)
        if individual_probs:
            avg_prob = np.mean(list(individual_probs.values()))
            std_prob = np.std(list(individual_probs.values()))
            # High average with low std = strong prediction
            strength_score = (avg_prob * 100) * (1 - min(std_prob, 0.5))
        else:
            strength_score = confidence_score
        
        # Component 4: Anomaly Indicators (0-100)
        anomaly_score = self._calculate_anomaly_score(feature_values) if feature_values else 0
        
        # Calculate weighted overall risk score
        overall_risk_score = (
            confidence_score * self.WEIGHTS['model_confidence'] +
            agreement_score * self.WEIGHTS['model_agreement'] +
            strength_score * self.WEIGHTS['prediction_strength'] +
            anomaly_score * self.WEIGHTS['anomaly_indicators']
        )
        
        # Determine risk level
        risk_level = self._get_risk_level(overall_risk_score)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(
            ensemble_prob, agreement_ratio if individual_preds else 0,
            individual_probs, feature_values
        )
        
        return {
            'overall_risk_score': round(overall_risk_score, 2),
            'risk_level': risk_level,
            'component_scores': {
                'model_confidence': round(confidence_score, 2),
                'model_agreement': round(agreement_score, 2),
                'prediction_strength': round(strength_score, 2),
                'anomaly_indicators': round(anomaly_score, 2)
            },
            'ensemble_probability': round(ensemble_prob, 4),
            'model_agreement_ratio': round(agreement_ratio if individual_preds else 0, 2),
            'risk_factors': risk_factors,
            'requires_investigation': overall_risk_score >= 80
        }
    
    def calculate_batch_risk_scores(
        self,
        ensemble_probs: np.ndarray,
        individual_probs_dict: Dict[str, np.ndarray],
        individual_preds_dict: Dict[str, np.ndarray],
        features_df: pd.DataFrame = None
    ) -> List[Dict[str, Any]]:
        """
        Calculate risk scores for a batch of records
        
        Args:
            ensemble_probs: Array of ensemble probabilities
            individual_probs_dict: Dict of model_name -> prob array
            individual_preds_dict: Dict of model_name -> pred array
            features_df: Optional DataFrame with feature values
            
        Returns:
            List of risk assessment dictionaries
        """
        self.log(f"Calculating risk scores for {len(ensemble_probs)} records...")
        
        risk_scores = []
        for i in range(len(ensemble_probs)):
            # Get individual model results for this record
            ind_probs = {name: probs[i] for name, probs in individual_probs_dict.items()}
            ind_preds = {name: preds[i] for name, preds in individual_preds_dict.items()}
            
            # Get feature values if available
            feature_vals = features_df.iloc[i].to_dict() if features_df is not None else None
            
            # Calculate risk score
            risk_data = self.calculate_risk_score(
                ensemble_prob=ensemble_probs[i],
                individual_probs=ind_probs,
                individual_preds=ind_preds,
                feature_values=feature_vals
            )
            
            risk_scores.append(risk_data)
        
        self.log(f"Completed risk scoring for {len(risk_scores)} records")
        return risk_scores
    
    def _calculate_anomaly_score(self, feature_values: Dict[str, float]) -> float:
        """
        Calculate anomaly score based on feature values
        Simple heuristic-based approach
        """
        if not feature_values:
            return 0.0
        
        anomaly_score = 0.0
        anomaly_count = 0
        
        # Check for extreme values (simple z-score approximation)
        values = [v for v in feature_values.values() if isinstance(v, (int, float)) and not np.isnan(v)]
        
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values) + 1e-8
            
            for val in values:
                z_score = abs((val - mean_val) / std_val)
                if z_score > 3:  # Outlier
                    anomaly_count += 1
            
            # Anomaly ratio
            anomaly_ratio = anomaly_count / len(values)
            anomaly_score = min(anomaly_ratio * 100, 100)
        
        return anomaly_score
    
    def _get_risk_level(self, score: float) -> str:
        """Categorize risk score into levels"""
        if score >= 80:
            return 'CRITICAL'
        elif score >= 60:
            return 'HIGH'
        elif score >= 40:
            return 'MEDIUM'
        elif score >= 20:
            return 'LOW'
        else:
            return 'MINIMAL'
    
    def _identify_risk_factors(
        self,
        ensemble_prob: float,
        agreement_ratio: float,
        individual_probs: Dict[str, float],
        feature_values: Dict[str, float]
    ) -> List[str]:
        """Identify specific risk factors"""
        factors = []
        
        # High confidence
        if ensemble_prob >= 0.8:
            factors.append(f"Very high fraud probability: {ensemble_prob:.2%}")
        elif ensemble_prob >= 0.6:
            factors.append(f"High fraud probability: {ensemble_prob:.2%}")
        
        # Strong model agreement
        if agreement_ratio >= 0.8:
            factors.append(f"Strong model consensus: {agreement_ratio:.0%} agreement")
        elif agreement_ratio >= 0.6:
            factors.append(f"Moderate model consensus: {agreement_ratio:.0%} agreement")
        
        # Individual model concerns
        if individual_probs:
            high_conf_models = [name for name, prob in individual_probs.items() if prob >= 0.8]
            if len(high_conf_models) >= 3:
                factors.append(f"{len(high_conf_models)} models show high confidence")
        
        # Feature anomalies
        if feature_values:
            anomaly_score = self._calculate_anomaly_score(feature_values)
            if anomaly_score >= 50:
                factors.append(f"Significant feature anomalies detected")
        
        return factors if factors else ["Standard risk assessment"]
    
    def get_risk_summary(self, risk_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for a batch of risk scores
        
        Args:
            risk_scores: List of risk assessment dictionaries
            
        Returns:
            Summary statistics dictionary
        """
        if not risk_scores:
            return {
                'total_records': 0,
                'message': 'No risk scores available'
            }
        
        scores = [r['overall_risk_score'] for r in risk_scores]
        levels = [r['risk_level'] for r in risk_scores]
        
        # Count by risk level
        level_counts = {}
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
            level_counts[level] = levels.count(level)
        
        # High-risk records
        high_risk = [
            {
                'index': i,
                'risk_score': r['overall_risk_score'],
                'risk_level': r['risk_level'],
                'ensemble_prob': r['ensemble_probability'],
                'risk_factors': r['risk_factors']
            }
            for i, r in enumerate(risk_scores)
            if r['overall_risk_score'] >= 60
        ]
        
        return {
            'total_records': len(risk_scores),
            'average_risk_score': round(np.mean(scores), 2),
            'max_risk_score': round(max(scores), 2),
            'min_risk_score': round(min(scores), 2),
            'median_risk_score': round(np.median(scores), 2),
            'std_risk_score': round(np.std(scores), 2),
            'risk_level_distribution': level_counts,
            'high_risk_count': len(high_risk),
            'critical_count': level_counts['CRITICAL'],
            'high_risk_records': sorted(high_risk, key=lambda x: x['risk_score'], reverse=True)[:20]
        }


class StructuredOutputFormatter:
    """
    Format structured pipeline output for multiagent system
    """
    
    def __init__(self, output_dir: str = "output"):
        """Initialize formatter"""
        import os
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def format_for_multiagent(
        self,
        record_index: int,
        features: Dict[str, Any],
        ensemble_prediction: int,
        ensemble_probability: float,
        individual_predictions: Dict[str, int],
        individual_probabilities: Dict[str, float],
        risk_data: Dict[str, Any],
        true_label: int = None
    ) -> Dict[str, Any]:
        """
        Format a single record for multiagent consumption
        
        Returns:
            Formatted dictionary ready for multiagent system
        """
        # Determine agent routing
        agents = self._determine_agent_routing(risk_data, ensemble_probability)
        
        return {
            'record_id': f"structured_{record_index}",
            'timestamp': datetime.now().isoformat(),
            'source_pipeline': 'structured',
            
            'prediction': {
                'ensemble_prediction': int(ensemble_prediction),
                'ensemble_probability': float(ensemble_probability),
                'individual_predictions': {k: int(v) for k, v in individual_predictions.items()},
                'individual_probabilities': {k: float(v) for k, v in individual_probabilities.items()},
                'true_label': int(true_label) if true_label is not None else None
            },
            
            'risk_assessment': risk_data,
            
            'features': {k: float(v) if isinstance(v, (int, float, np.number)) else str(v) 
                        for k, v in features.items()},
            
            'agent_routing': agents,
            
            'metadata': {
                'pipeline_type': 'structured',
                'data_type': 'tabular',
                'model_count': len(individual_predictions)
            }
        }
    
    def _determine_agent_routing(
        self,
        risk_data: Dict[str, Any],
        ensemble_prob: float
    ) -> Dict[str, Any]:
        """Determine which agents should process this record"""
        agents = []
        priority = 'normal'
        hints = []
        
        risk_score = risk_data['overall_risk_score']
        risk_level = risk_data['risk_level']
        
        # Route based on risk level
        if risk_score >= 80:
            agents.extend(['fraud_investigation_agent', 'alert_agent'])
            priority = 'critical'
            hints.append('immediate_review_required')
        elif risk_score >= 60:
            agents.extend(['fraud_detection_agent', 'risk_assessment_agent'])
            priority = 'high'
            hints.append('priority_review')
        elif risk_score >= 40:
            agents.append('risk_assessment_agent')
            priority = 'normal'
        else:
            agents.append('general_analysis_agent')
            priority = 'low'
        
        # Add statistical analysis agent for all
        agents.append('statistical_analysis_agent')
        
        # Model agreement hints
        if risk_data.get('model_agreement_ratio', 0) >= 0.8:
            hints.append('high_model_consensus')
        elif risk_data.get('model_agreement_ratio', 0) <= 0.4:
            hints.append('low_model_consensus_needs_review')
        
        return {
            'recommended_agents': list(set(agents)),
            'priority': priority,
            'processing_hints': hints
        }
    
    def save_batch_output(
        self,
        formatted_records: List[Dict[str, Any]],
        batch_name: str = "structured_analysis"
    ) -> str:
        """Save formatted batch output to JSON"""
        import json
        from pathlib import Path
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{batch_name}_{timestamp}.json"
        filepath = Path(self.output_dir) / filename
        
        output = {
            'batch_metadata': {
                'batch_name': batch_name,
                'timestamp': datetime.now().isoformat(),
                'total_records': len(formatted_records),
                'pipeline': 'structured'
            },
            'records': formatted_records
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        return str(filepath)
