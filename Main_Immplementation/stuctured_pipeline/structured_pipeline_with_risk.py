# File: Pipelines/stuctured_pipeline/structured_pipeline_with_risk.py
"""
Enhanced Structured Pipeline with Risk Scoring
Integrates risk assessment and multiagent output formatting
"""
import os
import sys
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

# Import risk scoring components
from risk_scorer_structured import StructuredRiskScorer, StructuredOutputFormatter

# Import original structured pipeline functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from structured_pipeline import (
    load_models, predict_single, load_data,
    MODEL_TYPES, MODELS_DIR, INPUT_CSV
)


class EnhancedStructuredPipeline:
    """
    Enhanced pipeline with risk scoring and multiagent integration
    """
    
    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        enable_risk_scoring: bool = True,
        enable_output_formatting: bool = True
    ):
        """
        Initialize enhanced pipeline
        
        Args:
            models_dir: Directory containing trained models
            enable_risk_scoring: Whether to calculate risk scores
            enable_output_formatting: Whether to format output for multiagent system
        """
        self.models_dir = models_dir
        self.enable_risk_scoring = enable_risk_scoring
        self.enable_output_formatting = enable_output_formatting
        
        # Initialize components
        self.risk_scorer = StructuredRiskScorer() if enable_risk_scoring else None
        self.output_formatter = StructuredOutputFormatter() if enable_output_formatting else None
        
        # Load models
        print("=" * 80)
        print("ENHANCED STRUCTURED PIPELINE WITH RISK SCORING")
        print("=" * 80)
        self.models, self.scalers, self.metadata = load_models()
        
        if not self.models:
            raise ValueError("No models loaded!")
        
        print(f"\nRisk Scoring: {'Enabled' if enable_risk_scoring else 'Disabled'}")
        print(f"Output Formatting: {'Enabled' if enable_output_formatting else 'Disabled'}")
    
    def run(
        self,
        input_csv: str = INPUT_CSV,
        export_output: bool = True,
        batch_name: str = "structured_risk_analysis"
    ) -> dict:
        """
        Run the enhanced pipeline
        
        Args:
            input_csv: Path to input CSV file
            export_output: Whether to export formatted output
            batch_name: Name for the batch export
            
        Returns:
            Dictionary with results and statistics
        """
        print(f"\n{'=' * 80}")
        print("LOADING DATA")
        print("=" * 80)
        
        # Load and prepare data
        df, target_col = load_data(input_csv)
        print(f"Loaded {len(df)} records")
        
        # Prepare features
        sample_scaler = next(iter(self.scalers.values()))
        try:
            train_feats = sample_scaler.feature_names_in_
        except AttributeError:
            train_feats = [col for col in df.columns if col != target_col]
        
        X_df = df.drop(columns=[target_col], errors='ignore')
        X_df = X_df.reindex(columns=train_feats, fill_value=0)
        print(f"Feature shape: {X_df.shape}")
        
        # Get true labels if available
        y_true = df[target_col].values if target_col in df.columns else None
        
        # Run ensemble predictions
        print(f"\n{'=' * 80}")
        print("RUNNING ENSEMBLE PREDICTIONS")
        print("=" * 80)
        
        ensemble_pred, ensemble_prob, individual_preds, individual_probs = self._ensemble_predict(X_df)
        
        print(f"\nPredictions complete:")
        print(f"  Fraud cases: {ensemble_pred.sum()}/{len(ensemble_pred)} ({ensemble_pred.mean()*100:.2f}%)")
        
        # Calculate risk scores
        risk_scores = None
        if self.enable_risk_scoring:
            print(f"\n{'=' * 80}")
            print("CALCULATING RISK SCORES")
            print("=" * 80)
            
            risk_scores = self.risk_scorer.calculate_batch_risk_scores(
                ensemble_probs=ensemble_prob,
                individual_probs_dict=individual_probs,
                individual_preds_dict=individual_preds,
                features_df=X_df
            )
            
            # Display risk summary
            risk_summary = self.risk_scorer.get_risk_summary(risk_scores)
            self._display_risk_summary(risk_summary)
        
        # Format output for multiagent system
        formatted_records = None
        output_path = None
        
        if self.enable_output_formatting and export_output:
            print(f"\n{'=' * 80}")
            print("FORMATTING OUTPUT FOR MULTIAGENT SYSTEM")
            print("=" * 80)
            
            formatted_records = []
            for i in range(len(X_df)):
                # Get individual model results for this record
                ind_preds = {name: preds[i] for name, preds in individual_preds.items()}
                ind_probs = {name: probs[i] for name, probs in individual_probs.items()}
                
                # Get risk data
                risk_data = risk_scores[i] if risk_scores else {}
                
                # Format record
                formatted = self.output_formatter.format_for_multiagent(
                    record_index=i,
                    features=X_df.iloc[i].to_dict(),
                    ensemble_prediction=ensemble_pred[i],
                    ensemble_probability=ensemble_prob[i],
                    individual_predictions=ind_preds,
                    individual_probabilities=ind_probs,
                    risk_data=risk_data,
                    true_label=y_true[i] if y_true is not None else None
                )
                formatted_records.append(formatted)
            
            # Save to file
            output_path = self.output_formatter.save_batch_output(
                formatted_records,
                batch_name=batch_name
            )
            
            print(f"\n✓ Output saved to: {output_path}")
        
        # Save predictions CSV
        predictions_df = self._create_predictions_dataframe(
            X_df, ensemble_pred, ensemble_prob, risk_scores, y_true
        )
        
        pred_path = f"{batch_name}_predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        print(f"✓ Predictions CSV saved to: {pred_path}")
        
        # Return results
        return {
            'success': True,
            'total_records': len(X_df),
            'fraud_predicted': int(ensemble_pred.sum()),
            'fraud_percentage': float(ensemble_pred.mean() * 100),
            'risk_summary': risk_summary if risk_scores else None,
            'output_path': output_path,
            'predictions_path': pred_path,
            'formatted_records': formatted_records
        }
    
    def _ensemble_predict(self, X_df: pd.DataFrame):
        """Run ensemble prediction on input data"""
        total_weight = sum(m['auc_roc'] for m in self.metadata.values()) or 1.0
        ensemble_prob = np.zeros(len(X_df))
        
        individual_preds = {}
        individual_probs = {}
        
        for name, model in self.models.items():
            print(f"  → {self.metadata[name]['display_name']}")
            pred, prob = predict_single(name, model, self.scalers[name], X_df)
            
            individual_preds[name] = pred
            individual_probs[name] = prob
            
            weight = self.metadata[name]['auc_roc']
            ensemble_prob += prob * weight
        
        ensemble_prob /= total_weight
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        
        return ensemble_pred, ensemble_prob, individual_preds, individual_probs
    
    def _display_risk_summary(self, summary: dict):
        """Display risk summary statistics"""
        print(f"\nRisk Score Summary:")
        print(f"  Total records: {summary['total_records']}")
        print(f"  Average risk score: {summary['average_risk_score']:.2f}")
        print(f"  Max risk score: {summary['max_risk_score']:.2f}")
        print(f"  Min risk score: {summary['min_risk_score']:.2f}")
        print(f"  Median risk score: {summary['median_risk_score']:.2f}")
        
        print(f"\nRisk Level Distribution:")
        for level, count in summary['risk_level_distribution'].items():
            pct = (count / summary['total_records'] * 100) if summary['total_records'] > 0 else 0
            print(f"  {level:12s}: {count:6d} ({pct:5.2f}%)")
        
        print(f"\nHigh-Risk Records: {summary['high_risk_count']}")
        print(f"Critical Records: {summary['critical_count']}")
        
        if summary['high_risk_records']:
            print(f"\nTop 10 High-Risk Records:")
            for i, record in enumerate(summary['high_risk_records'][:10], 1):
                print(f"  {i:2d}. Record #{record['index']:6d}: "
                      f"Risk={record['risk_score']:6.2f} ({record['risk_level']}), "
                      f"Prob={record['ensemble_prob']:.4f}")
    
    def _create_predictions_dataframe(
        self,
        X_df: pd.DataFrame,
        ensemble_pred: np.ndarray,
        ensemble_prob: np.ndarray,
        risk_scores: list = None,
        y_true: np.ndarray = None
    ) -> pd.DataFrame:
        """Create DataFrame with predictions and risk scores"""
        result = X_df.copy()
        result['fraud_predicted'] = ensemble_pred
        result['fraud_probability'] = ensemble_prob
        
        if risk_scores:
            result['risk_score'] = [r['overall_risk_score'] for r in risk_scores]
            result['risk_level'] = [r['risk_level'] for r in risk_scores]
            result['requires_investigation'] = [r['requires_investigation'] for r in risk_scores]
        
        if y_true is not None:
            result['true_label'] = y_true
            result['correct_prediction'] = (ensemble_pred == y_true).astype(int)
        
        return result


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Enhanced Structured Pipeline with Risk Scoring'
    )
    parser.add_argument(
        '--input',
        type=str,
        default=INPUT_CSV,
        help='Input CSV file path'
    )
    parser.add_argument(
        '--export',
        action='store_true',
        help='Export formatted output for multiagent system'
    )
    parser.add_argument(
        '--batch-name',
        type=str,
        default='structured_risk_analysis',
        help='Batch name for output files'
    )
    parser.add_argument(
        '--disable-risk-scoring',
        action='store_true',
        help='Disable risk scoring (faster processing)'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = EnhancedStructuredPipeline(
            enable_risk_scoring=not args.disable_risk_scoring,
            enable_output_formatting=args.export
        )
        
        # Run pipeline
        result = pipeline.run(
            input_csv=args.input,
            export_output=args.export,
            batch_name=args.batch_name
        )
        
        # Display final summary
        print(f"\n{'=' * 80}")
        print("PIPELINE COMPLETE")
        print("=" * 80)
        print(f"✓ Processed {result['total_records']} records")
        print(f"✓ Fraud predicted: {result['fraud_predicted']} ({result['fraud_percentage']:.2f}%)")
        
        if result.get('output_path'):
            print(f"✓ Multiagent output: {result['output_path']}")
        
        print(f"✓ Predictions CSV: {result['predictions_path']}")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
