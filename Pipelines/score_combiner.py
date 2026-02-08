"""
Score Combiner Module
Combines risk scores from structured and unstructured pipelines
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.output_schema import UnifiedOutput, RiskAssessment, PipelineMetadata, BatchOutput
from shared.utils import load_config, setup_logging, ensure_directory


class ScoreCombiner:
    """
    Combines risk scores from multiple pipelines into a unified assessment
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize score combiner
        
        Args:
            config: Configuration dictionary (uses defaults if None)
        """
        self.config = config or load_config()
        self.logger = setup_logging(
            level=self.config.get('logging', {}).get('level', 'INFO')
        )
        
        # Score combination settings
        self.combination_config = self.config.get('score_combination', {})
        self.structured_weight = self.combination_config.get('structured_weight', 0.6)
        self.unstructured_weight = self.combination_config.get('unstructured_weight', 0.4)
        self.conflict_threshold = self.combination_config.get('conflict_threshold', 30)
        self.missing_penalty = self.combination_config.get('missing_penalty', 0.8)
        
        self.logger.info(f"ScoreCombiner initialized with weights: "
                        f"structured={self.structured_weight}, "
                        f"unstructured={self.unstructured_weight}")
    
    def combine_with_agents(
        self,
        structured_risk: Optional[RiskAssessment],
        unstructured_risk: Optional[RiskAssessment],
        agent_results: Optional[Dict[str, Any]] = None
    ) -> RiskAssessment:
        """
        Combine pipeline scores with agent scores
        
        Args:
            structured_risk: Risk from structured pipeline
            unstructured_risk: Risk from unstructured pipeline
            agent_results: Dictionary of agent results from orchestrator
        
        Returns:
            Combined risk assessment with all sources
        """
        # Start with pipeline combination
        pipeline_risk = self.combine_scores(structured_risk, unstructured_risk)
        
        if not agent_results or not agent_results.get('combined_score'):
            # No agent data, return pipeline risk only
            return pipeline_risk
        
        # Get agent score and weights
        agent_score = agent_results.get('combined_score', 0.0)
        agent_confidence = agent_results.get('confidence', 0.5)
        agent_weight_sum = sum(agent_results.get('weights_used', {}).values())
        
        # Calculate combined weight with dynamic adjustment
        # If agents available, combine pipeline + agent scores
        # Pipeline weight = structured_weight + unstructured_weight
        pipeline_weight = self.structured_weight + self.unstructured_weight
        
        # Normalize weights
        total_weight = pipeline_weight + agent_weight_sum
        if total_weight > 0:
            norm_pipeline_weight = pipeline_weight / total_weight
            norm_agent_weight = agent_weight_sum / total_weight
        else:
            norm_pipeline_weight = 1.0
            norm_agent_weight = 0.0
        
        # Combine scores
        final_score = (
            pipeline_risk.overall_risk_score * norm_pipeline_weight +
            agent_score * norm_agent_weight
        )
        
        # Combine confidence
        final_confidence = (
            pipeline_risk.confidence * norm_pipeline_weight +
            agent_confidence * norm_agent_weight
        )
        
        # Merge risk factors
        combined_factors = list(pipeline_risk.risk_factors or [])
        if agent_results.get('agents_succeeded', 0) > 0:
            combined_factors.insert(0, 
                f"✓ {agent_results['agents_succeeded']} additional agents analyzed"
            )
        
        # Build component scores
        component_scores = dict(pipeline_risk.component_scores or {})
        component_scores.update({
            'agent_score': agent_score,
            'agent_confidence': agent_confidence,
            'agents_succeeded': agent_results.get('agents_succeeded', 0),
            'final_combined_score': round(final_score, 2)
        })
        
        return RiskAssessment(
            overall_risk_score=round(final_score, 2),
            risk_level=self._categorize_risk_level(final_score),
            component_scores=component_scores,
            risk_factors=combined_factors,
            confidence=round(final_confidence, 2)
        )
    
    def combine_scores(
        self,
        structured_risk: Optional[RiskAssessment],
        unstructured_risk: Optional[RiskAssessment]
    ) -> RiskAssessment:
        """
        Combine risk scores from both pipelines
        
        Args:
            structured_risk: Risk assessment from structured pipeline
            unstructured_risk: Risk assessment from unstructured pipeline
        
        Returns:
            Combined risk assessment
        """
        # Case 1: Both available
        if structured_risk and unstructured_risk:
            return self._combine_both(structured_risk, unstructured_risk)
        
        # Case 2: Only structured available
        elif structured_risk and not unstructured_risk:
            return self._single_source(structured_risk, 'structured', self.missing_penalty)
        
        # Case 3: Only unstructured available
        elif unstructured_risk and not structured_risk:
            return self._single_source(unstructured_risk, 'unstructured', self.missing_penalty)
        
        # Case 4: Neither available (shouldn't happen)
        else:
            self.logger.warning("No risk assessments available for combining")
            return RiskAssessment(
                overall_risk_score=0.0,
                risk_level='UNKNOWN',
                component_scores={},
                risk_factors=['No risk data available'],
                confidence=0.0
            )
    
    def _combine_both(
        self,
        structured_risk: RiskAssessment,
        unstructured_risk: RiskAssessment
    ) -> RiskAssessment:
        """Combine risk scores when both are available"""
        
        # Calculate weighted average
        combined_score = (
            structured_risk.overall_risk_score * self.structured_weight +
            unstructured_risk.overall_risk_score * self.unstructured_weight
        )
        
        # Check for conflicts
        score_diff = abs(structured_risk.overall_risk_score - unstructured_risk.overall_risk_score)
        has_conflict = score_diff > self.conflict_threshold
        
        # Determine combined risk level (take the higher if conflict)
        if has_conflict:
            risk_level = self._higher_risk_level(
                structured_risk.risk_level,
                unstructured_risk.risk_level
            )
        else:
            risk_level = self._categorize_risk_level(combined_score)
        
        # Combine risk factors
        combined_factors = []
        if structured_risk.risk_factors:
            combined_factors.extend([f"[Structured] {f}" for f in structured_risk.risk_factors[:3]])
        if unstructured_risk.risk_factors:
            combined_factors.extend([f"[Unstructured] {f}" for f in unstructured_risk.risk_factors[:3]])
        
        if has_conflict:
            combined_factors.insert(0, f"⚠️ Score conflict detected: "
                                    f"Structured={structured_risk.overall_risk_score:.1f}, "
                                    f"Unstructured={unstructured_risk.overall_risk_score:.1f}")
        
        # Combine component scores
        component_scores = {
            'structured_score': structured_risk.overall_risk_score,
            'unstructured_score': unstructured_risk.overall_risk_score,
            'weighted_combined': combined_score,
            'score_difference': score_diff
        }
        
        # Calculate confidence (lower if conflict)
        confidence = 1.0 if not has_conflict else max(0.5, 1.0 - (score_diff / 100))
        
        return RiskAssessment(
            overall_risk_score=round(combined_score, 2),
            risk_level=risk_level,
            component_scores=component_scores,
            risk_factors=combined_factors,
            confidence=round(confidence, 2)
        )
    
    def _single_source(
        self,
        risk: RiskAssessment,
        source: str,
        penalty: float
    ) -> RiskAssessment:
        """Handle single-source risk assessment with confidence penalty"""
        
        penalized_score = risk.overall_risk_score * penalty
        
        # Add note about single source
        risk_factors = [f"⚠️ Only {source} pipeline data available"] + risk.risk_factors
        
        return RiskAssessment(
            overall_risk_score=round(penalized_score, 2),
            risk_level=self._categorize_risk_level(penalized_score),
            component_scores={f'{source}_score': risk.overall_risk_score},
            risk_factors=risk_factors,
            confidence=round(penalty, 2)
        )
    
    def _categorize_risk_level(self, score: float) -> str:
        """Categorize risk score into level"""
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
    
    def _higher_risk_level(self, level1: str, level2: str) -> str:
        """Return the higher of two risk levels"""
        risk_order = {'MINIMAL': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4, 'UNKNOWN': -1}
        
        order1 = risk_order.get(level1, -1)
        order2 = risk_order.get(level2, -1)
        
        return level1 if order1 >= order2 else level2
    
    def determine_agent_routing(self, combined_risk: RiskAssessment) -> Tuple[List[str], str, List[str]]:
        """
        Determine agent routing based on combined risk assessment
        
        Returns:
            Tuple of (recommended_agents, priority, processing_hints)
        """
        agents = []
        priority = 'normal'
        hints = []
        
        score = combined_risk.overall_risk_score
        level = combined_risk.risk_level
        
        # Route based on risk level
        if score >= 80:
            agents = ['fraud_investigation_agent', 'alert_agent', 'compliance_agent']
            priority = 'critical'
            hints = ['immediate_review_required', 'escalate_to_human']
        elif score >= 60:
            agents = ['fraud_detection_agent', 'risk_assessment_agent', 'compliance_agent']
            priority = 'high'
            hints = ['priority_review', 'detailed_analysis_needed']
        elif score >= 40:
            agents = ['risk_assessment_agent', 'pattern_analysis_agent']
            priority = 'normal'
            hints = ['standard_review']
        else:
            agents = ['general_analysis_agent', 'statistical_analysis_agent']
            priority = 'low'
            hints = ['routine_monitoring']
        
        # Add hints based on confidence
        if combined_risk.confidence < 0.7:
            hints.append('low_confidence_review_needed')
        
        # Check for conflicts in factors
        if any('conflict' in f.lower() for f in combined_risk.risk_factors):
            hints.append('score_conflict_detected')
            if priority == 'low':
                priority = 'normal'
        
        return agents, priority, hints
    
    def combine_batch(
        self,
        structured_output: Optional[str] = None,
        unstructured_output: Optional[str] = None,
        output_path: Optional[str] = None,
        batch_name: str = "combined_analysis",
        agent_results: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Combine batch outputs from both pipelines and agents
        
        Args:
            structured_output: Path to structured pipeline output JSON
            unstructured_output: Path to unstructured pipeline output JSON
            output_path: Path for combined output (auto-generated if None)
            batch_name: Name for the batch
            agent_results: Agent orchestrator results (optional)
        
        Returns:
            Path to combined output file
        """
        self.logger.info("=" * 80)
        self.logger.info("COMBINING PIPELINE OUTPUTS")
        self.logger.info("=" * 80)
        
        # Debug logging for agent results
        if agent_results:
            self.logger.info(f"Agent results received: keys={list(agent_results.keys())}")
            self.logger.info(f"  Combined score: {agent_results.get('combined_score')}")
            self.logger.info(f"  Individual results: {list(agent_results.get('individual_results', {}).keys())}")
        else:
            self.logger.info("No agent results received")
        
        # Load outputs
        structured_data = self._load_json(structured_output) if structured_output else None
        unstructured_data = self._load_json(unstructured_output) if unstructured_output else None
        
        if not structured_data and not unstructured_data:
            raise ValueError("At least one pipeline output must be provided")
        
        # Create batch output
        batch = BatchOutput(batch_name=batch_name)
        
        # Match and combine records
        matched_records = self._match_records(structured_data, unstructured_data)
        
        self.logger.info(f"Matched {len(matched_records)} records for combination")
        
        for record_data in matched_records:
            # Add agent results to record data
            record_data['agent_results'] = agent_results
            unified = self._create_unified_output(record_data)
            batch.add_record(unified)
        
        # Save combined output
        if output_path is None:
            output_dir = ensure_directory(self.config['output']['combined_dir'])
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = output_dir / f"{batch_name}_{timestamp}.json"
        
        batch.save(str(output_path))
        
        self.logger.info(f"✓ Combined output saved to: {output_path}")
        self.logger.info(f"  Total records: {len(batch.records)}")
        self.logger.info(f"  Critical: {batch.summary_stats.get('critical_count', 0)}")
        self.logger.info(f"  High risk: {batch.summary_stats.get('high_risk_count', 0)}")
        
        return str(output_path)
    
    def _load_json(self, path: str) -> Optional[Dict]:
        """Load JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load {path}: {e}")
            return None
    
    def _match_records(
        self,
        structured_data: Optional[Dict],
        unstructured_data: Optional[Dict]
    ) -> List[Dict[str, Any]]:
        """
        Match records from both pipelines
        
        Returns list of matched record dictionaries with keys:
        - record_id
        - structured (if available)
        - unstructured (if available)
        """
        matched = []
        
        # Extract records from both sources
        structured_records = {}
        if structured_data:
            # Handle both array format and single-result format
            if 'records' in structured_data:
                # Batch format with records array
                records = structured_data.get('records', [])
                for record in records:
                    rid = record.get('record_id', str(len(structured_records)))
                    structured_records[rid] = record
            elif 'input_file' in structured_data:
                # Single result format (structured pipeline output)
                # Use input_file name as record_id
                rid = structured_data.get('input_file', 'unknown')
                structured_records[rid] = structured_data
            else:
                # Generic fallback
                structured_records['result_0'] = structured_data
        
        unstructured_records = {}
        if unstructured_data:
            documents = unstructured_data.get('documents', [])
            for doc in documents:
                # Try to extract record_id
                rid = doc.get('record_id') or doc.get('doc_id') or doc.get('document', {}).get('doc_id')
                if rid:
                    unstructured_records[rid] = doc
        
        # Match records
        all_ids = set(structured_records.keys()) | set(unstructured_records.keys())
        
        for rid in all_ids:
            matched.append({
                'record_id': rid,
                'structured': structured_records.get(rid),
                'unstructured': unstructured_records.get(rid)
            })
        
        return matched
    
    def _create_unified_output(self, record_data: Dict[str, Any]) -> UnifiedOutput:
        """Create unified output from matched record data"""
        
        record_id = record_data['record_id']
        structured = record_data.get('structured')
        unstructured = record_data.get('unstructured')
        agent_results = record_data.get('agent_results')
        
        # Extract risk assessments
        structured_risk = self._extract_risk_assessment(structured, 'structured') if structured else None
        unstructured_risk = self._extract_risk_assessment(unstructured, 'unstructured') if unstructured else None
        
        # Combine scores (with agents if available)
        if agent_results and agent_results.get('combined_score'):
            combined_risk = self.combine_with_agents(structured_risk, unstructured_risk, agent_results)
        else:
            combined_risk = self.combine_scores(structured_risk, unstructured_risk)
        
        # Determine routing
        agents, priority, hints = self.determine_agent_routing(combined_risk)
        
        # Extract other fields
        fraud_prediction = structured.get('prediction', {}).get('ensemble_prediction') if structured else None
        fraud_probability = structured.get('prediction', {}).get('ensemble_probability') if structured else None
        
        entities = unstructured.get('entities') if unstructured else None
        relationships = unstructured.get('relationships') if unstructured else None
        
        # Prepare agent analysis results for output
        agent_analysis = None
        if agent_results and agent_results.get('individual_results'):
            agent_analysis = {
                'combined_score': agent_results.get('combined_score'),
                'confidence': agent_results.get('confidence'),
                'agents_executed': agent_results.get('agents_succeeded', 0),
                'agents_failed': agent_results.get('agents_failed', 0),
                'individual_agents': {}
            }
            
            # Add each agent's findings
            for agent_name, agent_result in agent_results.get('individual_results', {}).items():
                if agent_result.get('success'):
                    agent_analysis['individual_agents'][agent_name] = {
                        'score': agent_result.get('score'),
                        'confidence': agent_result.get('confidence'),
                        'findings': agent_result.get('findings', []),
                        'metrics': agent_result.get('metrics', {})
                    }
        
        # Create unified output
        unified = UnifiedOutput(
            record_id=record_id,
            source_identifier=unstructured.get('document', {}).get('file_name') if unstructured else None,
            structured_risk=structured_risk,
            unstructured_risk=unstructured_risk,
            combined_risk=combined_risk,
            fraud_prediction=fraud_prediction,
            fraud_probability=fraud_probability,
            entities=entities,
            relationships=relationships,
            recommended_agents=agents,
            priority=priority,
            processing_hints=hints,
            requires_investigation=combined_risk.overall_risk_score >= 80,
            metadata=PipelineMetadata(
                pipeline_type='combined',
                timestamp=combined_risk.risk_factors[-1] if combined_risk.risk_factors else None
            )
        )
        
        # Add agent analysis to the output immediately (must be done after __init__ but properly set)
        if agent_analysis:
            # Set extra_fields directly - this will be included in to_dict()
            if not hasattr(unified, '_extra_fields'):
                unified._extra_fields = {}
            unified._extra_fields['agent_analysis'] = agent_analysis
            self.logger.debug(f"Added agent_analysis to unified output: {list(agent_analysis.keys())}")
            self.logger.debug(f"_extra_fields now contains: {list(unified._extra_fields.keys())}")
        
        return unified
    
    def _extract_risk_assessment(
        self,
        data: Dict[str, Any],
        source: str
    ) -> Optional[RiskAssessment]:
        """Extract RiskAssessment from pipeline output"""
        
        if not data:
            return None
        
        # Try to find risk data - handle multiple formats
        risk_data = data.get('risk_assessment') or data.get('risk_data')
        
        if risk_data:
            # Nested format
            return RiskAssessment(
                overall_risk_score=risk_data.get('overall_risk_score', 0.0),
                risk_level=risk_data.get('risk_level', 'UNKNOWN'),
                component_scores=risk_data.get('component_scores', {}),
                risk_factors=risk_data.get('risk_factors', []),
                confidence=1.0
            )
        elif 'risk_score' in data:
            # Direct format (structured pipeline output)
            # Convert risk_score (0-1) to percentage (0-100) for consistency
            risk_score = data.get('risk_score', 0.0)
            if risk_score <= 1.0:
                risk_score = risk_score * 100
            
            # Build risk factors from model results
            risk_factors = []
            if 'models_predicting_fraud' in data:
                fraud_models = data.get('models_predicting_fraud', [])
                if fraud_models:
                    risk_factors.append(f"{len(fraud_models)} models flagged fraud: {', '.join(fraud_models[:3])}")
            
            if 'overall_prediction' in data:
                risk_factors.append(f"Overall prediction: {data['overall_prediction']}")
            
            return RiskAssessment(
                overall_risk_score=round(risk_score, 2),
                risk_level=data.get('risk_level', 'UNKNOWN'),
                component_scores={
                    'fraud_model_count': data.get('fraud_model_count', 0),
                    'total_models': data.get('total_models', 0)
                },
                risk_factors=risk_factors,
                confidence=1.0
            )
        else:
            return None


def main():
    """CLI for score combiner"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Combine pipeline outputs')
    parser.add_argument('--structured', type=str, help='Path to structured pipeline output JSON')
    parser.add_argument('--unstructured', type=str, help='Path to unstructured pipeline output JSON')
    parser.add_argument('--output', type=str, help='Output path for combined results')
    parser.add_argument('--batch-name', type=str, default='combined_analysis', help='Batch name')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    if not args.structured and not args.unstructured:
        print("Error: At least one of --structured or --unstructured must be provided")
        sys.exit(1)
    
    # Load config
    config = load_config(args.config) if args.config else load_config()
    
    # Combine scores
    combiner = ScoreCombiner(config)
    output_path = combiner.combine_batch(
        structured_output=args.structured,
        unstructured_output=args.unstructured,
        output_path=args.output,
        batch_name=args.batch_name
    )
    
    print(f"\n✅ Combined output saved to: {output_path}")


if __name__ == '__main__':
    main()
