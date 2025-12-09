# full_comparison_pipeline.py
# Runs: Clustering only | Classification only | Full Ensemble → Compares all 3

import os
import pandas as pd
import numpy as np
from jinja2 import Template
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc

# Reuse functions from above (we'll import logic inline to keep standalone)
from clustering_pipeline import run_clustering_ensemble
from classification_pipeline import run_classification_ensemble

# Reuse original full ensemble logic (slightly modified)
def run_full_ensemble():
    from structured_pipeline import main as run_original
    print("Running Full Ensemble (Clustering + Classification)...")
    run_original()
    df = pd.read_csv("ensemble_predictions.csv")
    return df['fraud_predicted'].values, df['fraud_confidence'].values, "Full Ensemble"

def load_data_for_metrics():
    df = pd.read_csv("../Dataset/enhanced_financial_data.csv", low_memory=False)
    for c in df.columns:
        if df[c].dtype == 'object':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    target = 'is_fraudulent' if 'is_fraudulent' in df.columns else 'misstate'
    if target not in df.columns:
        return df.drop(['fyear', 'gvkey'], axis=1, errors='ignore'), None
    y = df[target].dropna()
    X = df.loc[y.index].drop(columns=[target, 'fyear', 'gvkey'], errors='ignore')
    return X, y.values

def generate_comparison_report(results):
    X, y_true = load_data_for_metrics()

    metrics = []
    for name, pred, prob, _ in results:
        if y_true is None:
            metrics.append({
                'model': name,
                'accuracy': 'N/A',
                'precision': 'N/A',
                'recall': 'N/A',
                'f1': 'N/A',
                'auc': 'N/A',
                'ap': 'N/A',
                'flagged': int(pred.sum()),
                'rate': f"{pred.mean()*100:.2f}%"
            })
        else:
            metrics.append({
                'model': name,
                'accuracy': round(accuracy_score(y_true, pred), 4),
                'precision': round(precision_score(y_true, pred, zero_division=0), 4),
                'recall': round(recall_score(y_true, pred, zero_division=0), 4),
                'f1': round(f1_score(y_true, pred, zero_division=0), 4),
                'auc': round(roc_auc_score(y_true, prob), 4),
                'ap': round(average_precision_score(y_true, prob), 4),
                'flagged': int(pred.sum()),
                'rate': f"{pred.mean()*100:.2f}%"
            })

    # Plot ROC & PR
    if y_true is not None:
        plt.figure(figsize=(14, 6))

        plt.subplot(1, 2, 1)
        for name, _, prob, label in results:
            fpr, tpr, _ = roc_curve(y_true, prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{label} ({roc_auc:.3f})")
        plt.plot([0,1],[0,1],'k--')
        plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Comparison'); plt.legend()

        plt.subplot(1, 2, 2)
        for name, _, prob, label in results:
            p, r, _ = precision_recall_curve(y_true, prob)
            ap = average_precision_score(y_true, prob)
            plt.plot(r, p, label=f"{label} (AP={ap:.3f})")
        plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR Comparison'); plt.legend()

        plt.tight_layout()
        plt.savefig("final_comparison_plot.png", dpi=200)
        plt.close()

    # HTML Report
    template = """<!DOCTYPE html>
<html><head><title>3-Way Ensemble Comparison</title>
<style>body{font-family:Arial;margin:40px;line-height:1.6}
table{border-collapse:collapse;width:100%;margin:25px 0}
th,td{border:1px solid #ddd;padding:12px;text-align:center}
th{background:#2c3e50;color:white}
h1,h2{color:#2c3e50}
img{max-width:100%;margin:20px 0}
.highlight{background:#e8f5e9;font-weight:bold}</style></head>
<body>
<h1>Fraud Detection: 3-Way Ensemble Comparison</h1>
<p><b>Dataset:</b> {{total}} records | {% if y_true %}Labels available{% else %}No labels (unsupervised mode){% endif %}</p>
{% if plot %}<h2>ROC & Precision-Recall Curves</h2><img src="final_comparison_plot.png">{% endif %}

<h2>Performance Summary</h2>
<table>
<tr><th>Ensemble Type</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>AUC-ROC</th><th>Avg Precision</th><th>Flagged</th><th>Rate</th></tr>
{% for m in metrics %}
<tr {% if 'Full' in m.model %}class="highlight"{% endif %}>
<td><b>{{m.model}}</b></td>
<td>{{m.accuracy}}</td><td>{{m.precision}}</td><td>{{m.recall}}</td>
<td>{{m.f1}}</td><td>{{m.auc}}</td><td>{{m.ap}}</td>
<td>{{m.flagged}}</td><td>{{m.rate}}</td>
</tr>
{% endfor %}
</table>
<p><i>Full Ensemble = Clustering + Classification (weighted by AUC)</i></p>
</body></html>"""

    has_labels = y_true is not None
    html = Template(template).render(
        total=len(X), y_true=has_labels,
        metrics=metrics,
        plot="final_comparison_plot.png" if has_labels and os.path.exists("final_comparison_plot.png") else None
    )
    with open("FINAL_COMPARISON_REPORT.html", "w") as f:
        f.write(html)
    print("FINAL_COMPARISON_REPORT.html generated!")

def main():
    print("=== Running Clustering-Only Ensemble ===")
    clust_pred, clust_prob, clust_name = run_clustering_ensemble() or (None, None, None)

    print("\n=== Running Classification-Only Ensemble ===")
    class_pred, class_prob, class_name = run_classification_ensemble() or (None, None, None)

    print("\n=== Running Full Ensemble (Both) ===")
    full_pred, full_prob, full_name = run_full_ensemble()

    results = []
    if clust_pred is not None:
        results.append(("clustering", clust_pred, clust_prob, clust_name))
    if class_pred is not None:
        results.append(("classification", class_pred, class_prob, class_name))
    if full_pred is not None:
        results.append(("full", full_pred, full_prob, full_name))

    if results:
        generate_comparison_report(results)
    else:
        print("No models ran successfully.")

if __name__ == "__main__":
    main()