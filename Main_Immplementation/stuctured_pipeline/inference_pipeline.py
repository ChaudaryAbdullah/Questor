"""
Fraud Detection Inference Pipeline

Main inference pipeline that:
1. Loads JSON financial data
2. Transforms to features using json_to_features module
3. Loads all trained models from MyModels/
4. Runs ensemble prediction with configurable weights
5. Outputs detailed results including individual model predictions
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Any
from tensorflow.keras.models import load_model
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

# Import our transformation module
from json_to_features import transform_json_to_features


# Configuration
MODELS_DIR = "MyModels"
WEIGHTS_CONFIG = "model_weights.json"
OUTPUT_DIR = "Output"

MODEL_TYPES = {
    'xgboost': 'XGBClassifier',
    'random_forest': 'RandomForestClassifier',
    'dnn': 'keras',
    'autoencoder': 'keras',
    'lstm': 'keras',
    'cnn': 'keras',
    'logistic_regression': 'LogisticRegression',
    'svm': 'SVC',
    'decision_tree': 'DecisionTreeClassifier',
    'isolation_forest': 'IsolationForest',
    'dbscan': 'DBSCAN',
    'kmeans': 'KMeans',
    'gmm': 'GaussianMixture',
    'pca_anomaly': 'PCA',
    'lightgbm': 'LGBMClassifier',
    'catboost': 'CatBoostClassifier',
    'lof': 'LocalOutlierFactor',
    'oneclass_svm': 'OneClassSVM'
}


def load_model_weights(weights_path: str = WEIGHTS_CONFIG) -> Dict[str, float]:
    """Load model weights from configuration file."""
    if not os.path.exists(weights_path):
        print(f"⚠️  Weights file not found: {weights_path}")
        print("Using equal weights for all models.")
        return {}
    
    with open(weights_path, 'r') as f:
        weights = json.load(f)
    
    print(f"✅ Loaded weights for {len(weights)} models from {weights_path}")
    return weights


def load_all_models(models_dir: str = MODELS_DIR) -> Tuple[Dict, Dict, Dict]:
    """
    Load all trained models, scalers, and metadata from models directory.
    
    Returns:
        Tuple of (models_dict, scalers_dict, metadata_dict)
    """
    models, scalers, metadata = {}, {}, {}
    
    if not os.path.exists(models_dir):
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    for folder in os.listdir(models_dir):
        folder_path = os.path.join(models_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        # Find model name from folder name
        model_name = None
        for candidate in MODEL_TYPES.keys():
            if folder.startswith(candidate + '_'):
                model_name = candidate
                break
        
        if not model_name:
            print(f"⚠️  Skipping folder (no model match): {folder}")
            continue
        
        # Find model and scaler files
        model_path = scaler_path = metrics_path = None
        for f in os.listdir(folder_path):
            fp = os.path.join(folder_path, f)
            if f == 'model.pkl':
                model_path = fp
            elif f == 'model.h5':
                model_path = fp
            elif f == 'scaler.pkl':
                scaler_path = fp
            elif f == 'metrics.txt':
                metrics_path = fp
        
        if not (model_path and scaler_path):
            print(f"⚠️  Missing files in {folder} → model: {model_path}, scaler: {scaler_path}")
            continue
        
        # Load model and scaler
        try:
            if model_path.endswith('.h5'):
                model = load_model(model_path)
            else:
                model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"❌ Failed to load {model_name} from {folder}: {e}")
            continue
        
        # Load AUC from metrics (for default weighting)
        auc = 0.5
        if metrics_path and os.path.exists(metrics_path):
            try:
                with open(metrics_path) as f:
                    for line in f:
                        if 'auc_roc' in line.lower() or 'auc' in line.lower():
                            parts = line.split(':')
                            if len(parts) > 1:
                                auc = float(parts[1].strip())
                                break
            except:
                pass
        
        # Store
        models[model_name] = model
        scalers[model_name] = scaler
        metadata[model_name] = {
            'folder': folder,
            'auc_roc': auc,
            'display_name': model_name.replace('_', ' ').title()
        }
        print(f"✅ Loaded: {metadata[model_name]['display_name']} (Training AUC: {auc:.4f})")
    
    print(f"\n🎯 Total models loaded: {len(models)}")
    return models, scalers, metadata


def predict_single_model(model_name: str, model: Any, scaler: Any, X_df: pd.DataFrame) -> Tuple[int, float]:
    """
    Run prediction for a single model.
    
    Returns:
        Tuple of (prediction, probability)
        - prediction: 0 (normal) or 1 (fraud)
        - probability: fraud probability [0, 1]
    """
    X = X_df.copy()
    
    # Scale features
    try:
        X_scaled = scaler.transform(X)
    except Exception:
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(X)
    
    # Keras models
    if model_name in ['dnn', 'autoencoder', 'lstm', 'cnn']:
        if model_name == 'lstm':
            n_feat = X_scaled.shape[1]
            ts = 10
            if n_feat < ts:
                ts = 1
                X_r = X_scaled.reshape(-1, 1, n_feat)
            else:
                pad = ts - (n_feat % ts)
                X_p = np.hstack([X_scaled, np.zeros((X_scaled.shape[0], pad))])
                X_r = X_p.reshape(-1, ts, -1)
            prob = model.predict(X_r, verbose=0).flatten()[0]
        
        elif model_name == 'cnn':
            X_r = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            prob = model.predict(X_r, verbose=0).flatten()[0]
        
        elif model_name == 'autoencoder':
            reconstruction = model.predict(X_scaled, verbose=0)
            mse = np.mean(np.power(X_scaled - reconstruction, 2), axis=1)
            prob = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
            prob = prob[0] if len(prob) > 0 else 0.5
        
        else:  # dnn
            prob = model.predict(X_scaled, verbose=0).flatten()[0]
        
        pred = 1 if prob > 0.5 else 0
    
    # Supervised scikit-learn models
    elif model_name in ['xgboost', 'random_forest', 'logistic_regression', 
                        'svm', 'decision_tree', 'lightgbm', 'catboost']:
        prob = model.predict_proba(X_scaled)[0, 1]
        pred = model.predict(X_scaled)[0]
    
    # Unsupervised models
    elif model_name == 'isolation_forest':
        pred = model.predict(X_scaled)[0]
        pred = 1 if pred == -1 else 0
        scores = -model.score_samples(X_scaled)
        prob = (scores[0] - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    elif model_name == 'dbscan':
        from sklearn.cluster import DBSCAN as SklearnDBSCAN
        db = SklearnDBSCAN(eps=getattr(model, 'eps', 0.5),
                          min_samples=getattr(model, 'min_samples', 5))
        labels = db.fit_predict(X_scaled)
        pred = 1 if labels[0] == -1 else 0
        # Distance to nearest neighbor
        k = min(5, len(X_scaled))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        dist, _ = nbrs.kneighbors(X_scaled)
        prob = dist[0, 0] if k == 1 else dist[0, -1]
        prob = float(prob)  # Normalize if needed
    
    elif model_name == 'kmeans':
        dist = model.transform(X_scaled)
        d = np.min(dist[0])
        prob = float(d) / (dist.max() + 1e-8)
        thr = np.percentile(dist, 95)
        pred = 1 if d > thr else 0
    
    elif model_name == 'gmm':
        scores = model.score_samples(X_scaled)
        prob = -scores[0]
        prob = float(prob) / (abs(scores).max() + 1e-8)
        thr = np.percentile(scores, 5)
        pred = 1 if scores[0] < thr else 0
    
    elif model_name == 'pca_anomaly':
        X_pca = model.transform(X_scaled)
        X_recon = model.inverse_transform(X_pca)
        mse = np.mean(np.power(X_scaled - X_recon, 2), axis=1)
        prob = (mse[0] - mse.min()) / (mse.max() - mse.min() + 1e-8)
        thr = np.percentile(mse, 95)
        pred = 1 if mse[0] > thr else 0
    
    elif model_name == 'lof':
        # LOF uses negative outlier factor
        scores = model.fit(X_scaled).negative_outlier_factor_
        prob = -scores[0]
        prob = (prob - scores.min()) / (scores.max() - scores.min() + 1e-8)
        pred = 1 if model.predict(X_scaled)[0] == -1 else 0
    
    elif model_name == 'oneclass_svm':
        pred = model.predict(X_scaled)[0]
        pred = 1 if pred == -1 else 0
        scores = model.score_samples(X_scaled)
        prob = -scores[0]
        prob = (prob - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Ensure probability is in [0, 1]
    prob = float(np.clip(prob, 0, 1))
    pred = int(pred)
    
    return pred, prob


def run_ensemble_prediction(
    json_path: str,
    models: Dict,
    scalers: Dict,
    metadata: Dict,
    weights: Dict[str, float]
) -> Dict[str, Any]:
    """
    Run complete inference pipeline on a JSON file.
    
    Returns:
        Dictionary with all prediction results
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(json_path)}")
    print(f"{'='*60}")
    
    # Step 1: Transform JSON to features
    print("\n[1/3] Transforming JSON to features...")
    try:
        features_df = transform_json_to_features(json_path)
        print(f"✅ Extracted {features_df.shape[1]} features")
    except Exception as e:
        raise RuntimeError(f"Failed to transform JSON: {e}")
    
    # Step 2: Align features with training data
    print("\n[2/3] Aligning features with training schema...")
    sample_scaler = next(iter(scalers.values()))
    try:
        train_features = sample_scaler.feature_names_in_
    except AttributeError:
        # Fallback: use all current features
        train_features = features_df.columns.tolist()
    
    features_df = features_df.reindex(columns=train_features, fill_value=0)
    print(f"✅ Aligned to {features_df.shape[1]} training features")
    
    # Step 3: Run predictions through all models
    print("\n[3/3] Running ensemble prediction...")
    
    individual_results = {}
    weighted_prob_sum = 0.0
    total_weight = 0.0
    
    for model_name, model in models.items():
        scaler = scalers[model_name]
        weight = weights.get(model_name, metadata[model_name]['auc_roc'])
        
        try:
            pred, prob = predict_single_model(model_name, model, scaler, features_df)
            
            individual_results[model_name] = {
                'display_name': metadata[model_name]['display_name'],
                'prediction': 'FRAUD' if pred == 1 else 'NORMAL',
                'fraud_probability': round(prob, 4),
                'weight': round(weight, 4),
                'training_auc': round(metadata[model_name]['auc_roc'], 4)
            }
            
            weighted_prob_sum += prob * weight
            total_weight += weight
            
            status = "🚨 FRAUD" if pred == 1 else "✅ NORMAL"
            print(f"  {metadata[model_name]['display_name']:25} → {status} (prob={prob:.4f}, weight={weight:.4f})")
        
        except Exception as e:
            print(f"  ⚠️  {metadata[model_name]['display_name']:25} → ERROR: {e}")
            individual_results[model_name] = {
                'display_name': metadata[model_name]['display_name'],
                'prediction': 'ERROR',
                'fraud_probability': None,
                'weight': weight,
                'error': str(e)
            }
    
    # Calculate weighted ensemble score
    if total_weight > 0:
        ensemble_risk_score = weighted_prob_sum / total_weight
    else:
        ensemble_risk_score = 0.5  # Default if all models failed
    
    # Determine which models flagged as fraud
    models_fraud = [r['display_name'] for r in individual_results.values() 
                    if r['prediction'] == 'FRAUD']
    models_normal = [r['display_name'] for r in individual_results.values() 
                     if r['prediction'] == 'NORMAL']
    
    # Final result
    result = {
        'input_file': os.path.basename(json_path),
        'risk_score': round(ensemble_risk_score, 6),
        'risk_level': get_risk_level(ensemble_risk_score),
        'overall_prediction': 'FRAUD' if ensemble_risk_score > 0.5 else 'NORMAL',
        'models_predicting_fraud': models_fraud,
        'models_predicting_normal': models_normal,
        'fraud_model_count': len(models_fraud),
        'normal_model_count': len(models_normal),
        'total_models': len(individual_results),
        'individual_model_results': individual_results
    }
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ENSEMBLE RISK SCORE: {ensemble_risk_score:.6f} ({get_risk_level(ensemble_risk_score)})")
    print(f"OVERALL PREDICTION: {result['overall_prediction']}")
    print(f"Models flagging FRAUD: {len(models_fraud)}/{len(individual_results)}")
    print(f"{'='*60}\n")
    
    return result


def get_risk_level(score: float) -> str:
    """Convert risk score to risk level."""
    if score >= 0.8:
        return "CRITICAL"
    elif score >= 0.6:
        return "HIGH"
    elif score >= 0.4:
        return "MODERATE"
    elif score >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"


def save_results(result: Dict[str, Any], output_dir: str = OUTPUT_DIR):
    """Save prediction results to JSON and text summary."""
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = os.path.splitext(result['input_file'])[0]
    
    # Save JSON
    json_path = os.path.join(output_dir, f"{base_name}_results.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"✅ Results saved to: {json_path}")
    
    # Save text summary
    txt_path = os.path.join(output_dir, f"{base_name}_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("FRAUD DETECTION ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write(f"Input File: {result['input_file']}\n")
        f.write(f"Risk Score: {result['risk_score']:.6f}\n")
        f.write(f"Risk Level: {result['risk_level']}\n")
        f.write(f"Overall Prediction: {result['overall_prediction']}\n\n")
        f.write(f"Models Predicting FRAUD ({len(result['models_predicting_fraud'])}):\n")
        for model in result['models_predicting_fraud']:
            f.write(f"  - {model}\n")
        f.write(f"\nModels Predicting NORMAL ({len(result['models_predicting_normal'])}):\n")
        for model in result['models_predicting_normal']:
            f.write(f"  - {model}\n")
        f.write("\n" + "="*60 + "\n")
        f.write("INDIVIDUAL MODEL DETAILS\n")
        f.write("="*60 + "\n\n")
        for model_name, details in result['individual_model_results'].items():
            f.write(f"{details['display_name']}:\n")
            f.write(f"  Prediction: {details['prediction']}\n")
            f.write(f"  Fraud Probability: {details.get('fraud_probability', 'N/A')}\n")
            f.write(f"  Weight: {details['weight']}\n")
            f.write(f"  Training AUC: {details.get('training_auc', 'N/A')}\n")
            if 'error' in details:
                f.write(f"  Error: {details['error']}\n")
            f.write("\n")
    print(f"✅ Summary saved to: {txt_path}")


def main(json_path: str):
    """Main entry point for inference pipeline."""
    # Load models and weights
    print("Loading models and configuration...")
    models, scalers, metadata = load_all_models()
    weights = load_model_weights()
    
    if not models:
        print("❌ No models loaded. Exiting.")
        return {'success': False, 'error': 'No models loaded'}
    
    # Run prediction
    result = run_ensemble_prediction(json_path, models, scalers, metadata, weights)
    
    # Add success flag
    result['success'] = True
    
    # Save results
    save_results(result)
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference_pipeline.py <path_to_json_file>")
        print("\nExample:")
        print("  python inference_pipeline.py Input/0000050493-2.json")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    if not os.path.exists(json_path):
        print(f"❌ File not found: {json_path}")
        sys.exit(1)
    
    main(json_path)
