# --------------------------------------------------------------
#  structured_pipeline.py  –  Ensemble inference + full report
# --------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import joblib
import json
import subprocess
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from jinja2 import Template
import warnings
warnings.filterwarnings("ignore")

# ------------------------------- CONFIG -------------------------------
MODELS_DIR = "MyModels"
INPUT_CSV = "../../Data/enhanced_financial_data.csv"
OUTPUT_PREDICTIONS = "ensemble_predictions.csv"
OUTPUT_REPORT = "model_comparison_report.html"
INDIVIDUAL_METRICS = "individual_metrics.txt"
COMPARISON_PLOT = "comparison_plots.png"

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
    'pca_anomaly': 'PCA'
}
# ------------------------------------------------------------------

# -------------------------- LOAD MODELS ---------------------------
def load_models():
    models, scalers, metadata = {}, {}, {}

    for folder in os.listdir(MODELS_DIR):
        folder_path = os.path.join(MODELS_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        # --- 1. Find the correct model name from MODEL_TYPES ---
        model_name = None
        for candidate in MODEL_TYPES.keys():
            if folder.startswith(candidate + '_'):
                model_name = candidate
                break

        if not model_name:
            print(f"Skipping folder (no model match): {folder}")
            continue

        # --- 2. Load files ---
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
            print(f"Missing files in {folder} → model: {model_path}, scaler: {scaler_path}")
            continue

        # --- 3. Load model & scaler ---
        try:
            model = load_model(model_path) if model_path.endswith('.h5') else joblib.load(model_path)
            scaler = joblib.load(scaler_path)
        except Exception as e:
            print(f"Failed to load {model_name} from {folder}: {e}")
            continue

        # --- 4. Load AUC from metrics.txt ---
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

        # --- 5. Store ---
        models[model_name] = model
        scalers[model_name] = scaler
        metadata[model_name] = {
            'folder': folder,
            'auc_roc': auc,
            'display_name': model_name.replace('_', ' ').title()
        }
        print(f"Loaded: {metadata[model_name]['display_name']} (AUC: {auc:.4f})")

    return models, scalers, metadata
# -------------------------- SINGLE PREDICTION --------------------
def predict_single(model_name, model, scaler, X_df):
    X = X_df.copy()

    # ---- scaling (robust) ----
    try:
        X_scaled = scaler.transform(X)
    except Exception:
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_scaled = scaler.transform(X)

    # ---- Keras models -------------------------------------------------
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
            prob = model.predict(X_r, verbose=0).flatten()

        elif model_name == 'cnn':
            X_r = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            prob = model.predict(X_r, verbose=0).flatten()

        elif model_name == 'autoencoder':
            recallon = model.predict(X_scaled, verbose=0)
            mse = np.mean(np.power(X_scaled - recallon, 2), axis=1)
            prob = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)

        else:   # dnn
            prob = model.predict(X_scaled, verbose=0).flatten()

        pred = (prob > 0.5).astype(int)

    # ---- scikit-learn supervised ------------------------------------
    elif model_name in ['xgboost', 'random_forest', 'logistic_regression',
                        'svm', 'decision_tree']:
        prob = model.predict_proba(X_scaled)[:, 1]
        pred = model.predict(X_scaled)

    # ---- unsupervised ------------------------------------------------
    elif model_name == 'isolation_forest':
        pred = model.predict(X_scaled)
        pred = np.where(pred == -1, 1, 0)
        scores = -model.score_samples(X_scaled)
        prob = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    elif model_name == 'dbscan':
        from sklearn.cluster import DBSCAN as SklearnDBSCAN
        db = SklearnDBSCAN(eps=getattr(model, 'eps', 0.5),
                           min_samples=getattr(model, 'min_samples', 5))
        labels = db.fit_predict(X_scaled)
        pred = np.where(labels == -1, 1, 0)
        # distance to nearest neighbour (max 5)
        k = min(5, len(X_scaled))
        nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
        dist, _ = nbrs.kneighbors(X_scaled)
        prob = dist[:, 0] if k == 1 else dist[:, -1]
        prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)

    elif model_name == 'kmeans':
        dist = model.transform(X_scaled)
        d = np.min(dist, axis=1)
        prob = (d - d.min()) / (d.max() - d.min() + 1e-8)
        thr = np.percentile(d, 95)
        pred = (d > thr).astype(int)

    elif model_name == 'gmm':
        scores = model.score_samples(X_scaled)
        prob = -scores
        prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)
        thr = np.percentile(scores, 5)
        pred = (scores < thr).astype(int)

    elif model_name == 'pca_anomaly':
        X_pca = model.transform(X_scaled)
        X_recall = model.inverse_transform(X_pca)
        mse = np.mean(np.power(X_scaled - X_recall, 2), axis=1)
        prob = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
        thr = np.percentile(mse, 95)
        pred = (mse > thr).astype(int)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    if len(prob) != len(X_df):
        raise ValueError(f"{model_name} gave {len(prob)} preds, expected {len(X_df)}")
    return pred, prob

# -------------------------- ENSEMBLE -----------------------------
def ensemble_predict(models, scalers, metadata, X_df):
    print(f"Predicting on {len(X_df)} input records...")
    total_weight = sum(m['auc_roc'] for m in metadata.values()) or 1.0
    ensemble_prob = np.zeros(len(X_df))

    preds, probs = {}, {}

    for name, model in models.items():
        print(f"  → {metadata[name]['display_name']} ({len(X_df)} samples)")
        pred, prob = predict_single(name, model, scalers[name], X_df)
        preds[name] = pred
        probs[name] = prob
        weight = metadata[name]['auc_roc']
        ensemble_prob += prob * weight

    ensemble_prob /= total_weight
    ensemble_pred = (ensemble_prob > 0.5).astype(int)
    
    # Return only predictions — metrics computed later
    return ensemble_pred, ensemble_prob, preds, probs

# -------------------------- LOAD & CLEAN -------------------------
def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Initial shape: {df.shape}")

    # object → numeric
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    print(f"After numeric conversion: {df.shape}")

    # target column
    target = 'is_fraudulent' if 'is_fraudulent' in df.columns else 'misstate'
    if target not in df.columns:
        raise ValueError("Target column missing")
    print(f"Target column: {target}")

    # drop rows with missing target (training only)
    df = df.dropna(subset=[target])
    print(f"After dropping NaN target: {df.shape}")

    # inf → NaN → 0
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    df = df.drop(['fyear', 'gvkey'], axis=1, errors='ignore')
    print(f"Final cleaned shape: {df.shape}")
    return df, target

# -------------------------- REPORT -------------------------------
def generate_report(X_df, y_true, preds, probs, ensemble_pred, ensemble_prob, metadata):
    # ---- Only compute metrics if labels exist ----
    if y_true is None:
        print("No true labels → skipping metrics & plots")
        # Save predictions only
        with open(INDIVIDUAL_METRICS, 'w') as f:
            f.write("No labels provided. Only AUC from training available.\n")
        print(f"Individual metrics → {INDIVIDUAL_METRICS}")
        return

    print("Computing real performance metrics on provided labels...")

    # ---- Helper to compute metrics ----
    def calc_metrics(y, pred, prob):
        return {
            'accuracy': accuracy_score(y, pred),
            'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0),
            'f1': f1_score(y, pred, zero_division=0),
            'auc': roc_auc_score(y, prob),
            'ap': average_precision_score(y, prob)
        }

    # ---- Individual model metrics ----
    indiv_metrics = []
    with open(INDIVIDUAL_METRICS, 'w') as f:
        f.write("Model\tAUC-ROC\tAccuracy\tPrecision\tRecall\tF1\tAvgPrecision\n")
        for name, prob in probs.items():
            pred = preds[name]
            m = calc_metrics(y_true, pred, prob)
            display = metadata[name]['display_name']
            f.write(f"{display}\t{m['auc']:.4f}\t{m['accuracy']:.4f}\t"
                    f"{m['precision']:.4f}\t{m['recall']:.4f}\t{m['f1']:.4f}\t{m['ap']:.4f}\n")
            indiv_metrics.append({
                'model': display,
                'accuracy': m['accuracy'],
                'precision': m['precision'],
                'recall': m['recall'],
                'f1': m['f1'],
                'auc': m['auc'],
                'ap': m['ap']
            })

        # ---- Ensemble metrics ----
        ens = calc_metrics(y_true, ensemble_pred, ensemble_prob)
        f.write(f"Ensemble\t{ens['auc']:.4f}\t{ens['accuracy']:.4f}\t"
                f"{ens['precision']:.4f}\t{ens['recall']:.4f}\t"
                f"{ens['f1']:.4f}\t{ens['ap']:.4f}\n")
    print(f"Individual metrics → {INDIVIDUAL_METRICS}")

    # ---- ROC + PR Plot (only if labels) ----
    plt.figure(figsize=(13, 5))

    ax1 = plt.subplot(1, 2, 1)
    for name, prob in probs.items():
        fpr, tpr, _ = roc_curve(y_true, prob)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f"{metadata[name]['display_name']} ({roc_auc:.3f})")
    fpr, tpr, _ = roc_curve(y_true, ensemble_prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, 'k--', lw=2, label=f"Ensemble ({roc_auc:.3f})")
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR'); ax1.set_title('ROC Curves')
    ax1.legend(fontsize=8)

    ax2 = plt.subplot(1, 2, 2)
    for name, prob in probs.items():
        prec, rec, _ = precision_recall_curve(y_true, prob)
        ap = average_precision_score(y_true, prob)
        ax2.plot(rec, prec, label=f"{metadata[name]['display_name']} ({ap:.3f})")
    prec, rec, _ = precision_recall_curve(y_true, ensemble_prob)
    ap = average_precision_score(y_true, ensemble_prob)
    ax2.plot(rec, prec, 'k--', lw=2, label=f"Ensemble ({ap:.3f})")
    ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.set_title('PR Curves')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(COMPARISON_PLOT, dpi=180)
    plt.close()
    print(f"Comparison plot → {COMPARISON_PLOT}")

    # ---- HTML Report ----
    template_str = """<!DOCTYPE html>
<html><head><title>Fraud Ensemble Report</title>
<style>body{font-family:Arial;margin:40px}
table{border-collapse:collapse;width:100%;margin:20px 0}
th,td{border:1px solid #ccc;padding:10px;text-align:left}
th{background:#f2f2f2}
h1,h2{color:#2c3e50}
img{max-width:100%}</style></head><body>
<h1>Fraud Detection Ensemble Report</h1>
<p><b>Input:</b> {{input_file}} ({{total}} records)</p>
<p><b>Fraud flagged:</b> {{fraud_cnt}} ({{fraud_pct}}%)</p>
<h2>Performance Metrics</h2>
<table><tr><th>Model</th><th>Acc</th><th>Prec</th><th>Rec</th><th>F1</th><th>AUC</th><th>AP</th></tr>
{% for m in metrics %}
<tr><td><b>{{m.model}}</b></td>
<td>{{ "%.4f"|format(m.accuracy) }}</td>
<td>{{ "%.4f"|format(m.precision) }}</td>
<td>{{ "%.4f"|format(m.recall) }}</td>
<td>{{ "%.4f"|format(m.f1) }}</td>
<td>{{ "%.4f"|format(m.auc) }}</td>
<td>{{ "%.4f"|format(m.ap) }}</td></tr>
{% endfor %}
<tr style="background:#e8f5e9"><td><b>Ensemble</b></td>
<td>{{ "%.4f"|format(ens.accuracy) }}</td>
<td>{{ "%.4f"|format(ens.precision) }}</td>
<td>{{ "%.4f"|format(ens.recall) }}</td>
<td>{{ "%.4f"|format(ens.f1) }}</td>
<td>{{ "%.4f"|format(ens.auc) }}</td>
<td>{{ "%.4f"|format(ens.ap) }}</td></tr>
</table>
<h2>Comparison Plots</h2>
<img src="{{plot}}" />
<h2>First 20 Predictions</h2>
<table><tr><th>#</th><th>Fraud?</th><th>Confidence</th></tr>
{% for i in range(20) %}
<tr><td>{{i}}</td><td>{{ "YES" if ens_pred[i] else "NO" }}</td>
<td>{{ "%.4f"|format(ens_prob[i]) }}</td></tr>
{% endfor %}
</table></body></html>"""

    html = Template(template_str).render(
        input_file=os.path.basename(INPUT_CSV),
        total=len(X_df),
        fraud_cnt=int(ensemble_pred.sum()),
        fraud_pct="%.2f" % (ensemble_pred.mean()*100),
        metrics=indiv_metrics,
        ens=ens,
        plot=COMPARISON_PLOT,
        ens_pred=ensemble_pred,
        ens_prob=ensemble_prob
    )
    with open(OUTPUT_REPORT, 'w') as f:
        f.write(html)
    print(f"HTML report → {OUTPUT_REPORT}")
    
# ------------------------------ MAIN ------------------------------
def main():
    df, target_col = load_data(INPUT_CSV)
    print(f"Loaded {len(df)} records for inference")

    models, scalers, metadata = load_models()
    if not models:
        print("No models found – aborting")
        return

    # ---- align features with training ----
    sample_scaler = next(iter(scalers.values()))
    # Safe way — works on ALL scikit-learn versions
    try:
        train_feats = sample_scaler.feature_names_in_
    except AttributeError:
        train_feats = sample_scaler.get_feature_names_out()  # also works if fitted on DataFrame
    except:
        # Fallback: assume features are all columns except target
        train_feats = [col for col in df.columns if col != 'is_fradulent']
    X_df = df.drop(columns=[target_col], errors='ignore')
    X_df = X_df.reindex(columns=train_feats, fill_value=0)
    print(f"Final input shape: {X_df.shape}")

    # ---- ensemble ----
    ens_pred, ens_prob, preds, probs = ensemble_predict(models, scalers, metadata, X_df)

    # ---- save predictions ----
    out = X_df.copy()
    out['fraud_predicted'] = ens_pred
    out['fraud_confidence'] = ens_prob
    if target_col in df.columns:
        out['true_label'] = df[target_col]

    import time
    base_name = "ensemble_predictions"
    extension = ".csv"
    final_path = OUTPUT_PREDICTIONS

    counter = 1
    while True:
        try:
            out.to_csv(final_path, index=False)
            print(f"Predictions saved → {final_path}")
            break
        except PermissionError:
            final_path = f"{base_name}_{counter}{extension}"
            counter += 1
            print(f"File in use. Trying → {final_path}")
            time.sleep(1)
        except Exception as e:
            print(f"Save failed: {e}")
            break
    print(f"Predictions → {OUTPUT_PREDICTIONS}")
    print(f"Fraud cases: {ens_pred.sum()}/{len(ens_pred)}")

    # ---- report (labels optional) ----
    y_true = df[target_col].values if target_col in df.columns else None
    generate_report(X_df, y_true, preds, probs, ens_pred, ens_prob, metadata)

if __name__ == "__main__":
    main()