# clustering_pipeline.py
# Only unsupervised / anomaly detection models

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings("ignore")

MODELS_DIR = "MyModels"
INPUT_CSV = "../Dataset/enhanced_financial_data.csv"
OUTPUT_PREDICTIONS = "clustering_predictions.csv"

# === UPDATED: Added oneclass_svm and lof ===
UNSUPERVISED_MODELS = [
    'isolation_forest', 'dbscan', 'kmeans', 'gmm', 'pca_anomaly', 'autoencoder',
    'oneclass_svm', 'lof'         # ← NEW
]

def load_clustering_models():
    models, scalers, metadata = {}, {}, {}

    for folder in os.listdir(MODELS_DIR):
        folder_path = os.path.join(MODELS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        model_name = None
        for candidate in UNSUPERVISED_MODELS:
            if folder.startswith(candidate + '_'):
                model_name = candidate
                break
        if not model_name:
            continue

        model_path = scaler_path = None
        for f in os.listdir(folder_path):
            fp = os.path.join(folder_path, f)
            if f in ['model.pkl', 'model.h5']:
                model_path = fp
            elif f == 'scaler.pkl':
                scaler_path = fp

        if not (model_path and scaler_path):
            continue

        try:
            model = load_model(model_path) if model_path.endswith('.h5') else joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        models[model_name] = model
        scalers[model_name] = scaler
        metadata[model_name] = {'display_name': model_name.replace('_', ' ').title()}
        print(f"Loaded clustering model: {metadata[model_name]['display_name']}")

    return models, scalers, metadata

def predict_single_unsupervised(model_name, model, scaler, X_df):
    X = X_df.copy()
    try:
        X_scaled = scaler.transform(X)
    except:
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(X)

    # Existing models...
    if model_name == 'autoencoder':
        recon = model.predict(X_scaled, verbose=0)
        mse = np.mean(np.power(X_scaled - recon, 2), axis=1)
        prob = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
        pred = (prob > np.percentile(prob, 95)).astype(int)

    elif model_name == 'isolation_forest':
        pred_raw = model.predict(X_scaled)
        pred = np.where(pred_raw == -1, 1, 0)
        scores = -model.score_samples(X_scaled)
        prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    elif model_name == 'dbscan':
        from sklearn.cluster import DBSCAN
        db = DBSCAN(eps=getattr(model, 'eps', 0.5), min_samples=getattr(model, 'min_samples', 5))
        labels = db.fit_predict(X_scaled)
        pred = np.where(labels == -1, 1, 0)
        k = min(5, len(X_scaled)) or 1
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        dist, _ = nbrs.kneighbors(X_scaled)
        prob = dist[:, -1] if k > 1 else dist[:, 0]
        prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)

    elif model_name == 'kmeans':
        dist = model.transform(X_scaled)
        d = np.min(dist, axis=1)
        prob = (d - d.min()) / (d.max() - d.min() + 1e-8)
        pred = (d > np.percentile(d, 95)).astype(int)

    elif model_name == 'gmm':
        scores = model.score_samples(X_scaled)
        prob = -scores
        prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
        pred = (scores < np.percentile(scores, 5)).astype(int)

    elif model_name == 'pca_anomaly':
        X_pca = model.transform(X_scaled)
        X_recon = model.inverse_transform(X_pca)
        mse = np.mean(np.power(X_scaled - X_recon, 2), axis=1)
        prob = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
        pred = (mse > np.percentile(mse, 95)).astype(int)

    # === NEW: One-Class SVM ===
    elif model_name == 'oneclass_svm':
        scores = model.decision_function(X_scaled)
        prob = -scores
        prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
        pred = model.predict(X_scaled)
        pred = np.where(pred == -1, 1, 0)  # -1 = anomaly

    # === NEW: Local Outlier Factor (LOF) ===
    elif model_name == 'lof':
        scores = model.decision_function(X_scaled)
        prob = -scores
        prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
        pred = model.predict(X_scaled)
        pred = np.where(pred == -1, 1, 0)

    return pred, prob

# run_clustering_ensemble() remains unchanged — it will now include the new models automatically

def run_clustering_ensemble():
    df = pd.read_csv(INPUT_CSV, low_memory=False)
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.drop(['fyear', 'gvkey'], axis=1, errors='ignore')

    models, scalers, metadata = load_clustering_models()
    if not models:
        print("No clustering models found!")
        return None, None, None

    sample_scaler = next(iter(scalers.values()))
    feature_cols = sample_scaler.feature_names_in_
    X = df.reindex(columns=feature_cols, fill_value=0)

    probs_list = []
    weights = []

    for name, model in models.items():
        pred, prob = predict_single_unsupervised(name, model, scalers[name], X)
        probs_list.append(prob)
        weights.append(1.0)  # Equal weight for unsupervised

    ensemble_prob = np.average(probs_list, axis=0, weights=weights)
    ensemble_pred = (ensemble_prob > 0.5).astype(int)

    result_df = X.copy()
    result_df['fraud_predicted'] = ensemble_pred
    result_df['fraud_confidence'] = ensemble_prob
    result_df.to_csv(OUTPUT_PREDICTIONS, index=False)
    print(f"Clustering ensemble saved → {OUTPUT_PREDICTIONS}")

    with open("clustering_summary.txt", "w") as f:
        f.write(f"Clustering Ensemble (Unsupervised Only)\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Fraud flagged: {ensemble_pred.sum()} ({ensemble_pred.mean()*100:.2f}%)\n")

    return ensemble_pred, ensemble_prob, "Clustering Ensemble"

if __name__ == "__main__":
    run_clustering_ensemble()