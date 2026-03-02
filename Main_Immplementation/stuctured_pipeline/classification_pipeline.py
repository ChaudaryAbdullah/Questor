# classification_pipeline.py
# Only supervised classification models

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = "MyModels"
INPUT_CSV = "../Dataset/enhanced_financial_data.csv"
OUTPUT_PREDICTIONS = "classification_predictions.csv"

# === UPDATED: Added lightgbm and catboost ===
SUPERVISED_MODELS = [
    'xgboost', 'random_forest', 'logistic_regression',
    'svm', 'decision_tree', 'dnn', 'lstm', 'cnn',
    'lightgbm', 'catboost'        # ← NEW
]

def load_classification_models():
    models, scalers, metadata = {}, {}, {}

    for folder in os.listdir(MODELS_DIR):
        folder_path = os.path.join(MODELS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        model_name = None
        for candidate in SUPERVISED_MODELS:
            if folder.startswith(candidate + '_'):
                model_name = candidate
                break
        if not model_name:
            continue

        model_path = scaler_path = metrics_path = None
        for f in os.listdir(folder_path):
            fp = os.path.join(folder_path, f)
            if f in ['model.pkl', 'model.h5']:
                model_path = fp
            elif f == 'scaler.pkl':
                scaler_path = fp
            elif f == 'metrics.txt':
                metrics_path = fp

        if not (model_path and scaler_path):
            continue

        try:
            model = load_model(model_path) if model_path.endswith('.h5') else joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Load failed {model_name}: {e}")
            continue

        auc = 0.5
        if metrics_path:
            try:
                with open(metrics_path) as f:
                    for line in f:
                        if 'auc_roc' in line.lower():
                            auc = float(line.split(':')[1].strip())
                            break
            except:
                pass

        models[model_name] = model
        scalers[model_name] = scaler
        metadata[model_name] = {
            'auc_roc': auc,
            'display_name': model_name.replace('_', ' ').title()
        }
        print(f"Loaded: {metadata[model_name]['display_name']} (AUC: {auc:.4f})")

    return models, scalers, metadata

# predict_single_supervised() remains unchanged — LightGBM & CatBoost use predict_proba() → works perfectly

def predict_single_supervised(model_name, model, scaler, X_df):
    X = X_df.copy()
    try:
        X_scaled = scaler.transform(X)
    except:
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(X)

    if model_name in ['dnn', 'lstm', 'cnn']:
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
            prob = model.predict(X_r, verbose=0).flatten()
        elif model_name == 'cnn':
            X_r = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            prob = model.predict(X_r, verbose=0).flatten()
        else:
            prob = model.predict(X_scaled, verbose=0).flatten()
        pred = (prob > 0.5).astype(int)
    else:
        # ← LightGBM, CatBoost, XGBoost, etc. all use this path
        prob = model.predict_proba(X_scaled)[:, 1]
        pred = model.predict(X_scaled)

    return pred, prob

def run_classification_ensemble():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.drop(['fyear', 'gvkey'], axis=1, errors='ignore')

    models, scalers, metadata = load_classification_models()
    if not models:
        print("No classification models found!")
        return None, None, None

    sample_scaler = next(iter(scalers.values()))
    feature_cols = sample_scaler.feature_names_in_
    X = df.reindex(columns=feature_cols, fill_value=0)

    total_weight = sum(m['auc_roc'] for m in metadata.values()) or 1.0
    ensemble_prob = np.zeros(len(X))

    for name, model in models.items():
        pred, prob = predict_single_supervised(name, model, scalers[name], X)
        weight = metadata[name]['auc_roc']
        ensemble_prob += prob * weight

    ensemble_prob /= total_weight
    ensemble_pred = (ensemble_prob > 0.5).astype(int)

    result_df = X.copy()
    result_df['fraud_predicted'] = ensemble_pred
    result_df['fraud_confidence'] = ensemble_prob
    result_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Classification ensemble saved → {OUTPUT_PREDICTIONS}")

    return ensemble_pred, ensemble_prob, "Classification Ensemble"

if __name__ == "__main__":
    run_classification_ensemble()