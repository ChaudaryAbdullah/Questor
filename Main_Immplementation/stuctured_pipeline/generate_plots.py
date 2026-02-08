import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np

data = """Model AUC-ROC Accuracy Precision Recall F1 AvgPrecision
Autoencoder 0.9208 0.9888 0.0000 0.0000 0.0000 0.1074
Cnn 0.9986 0.9955 0.7170 0.9856 0.8301 0.9465
Dbscan 0.6502 0.5192 0.0163 0.7032 0.0318 0.0459
Decision Tree 0.9967 0.9967 0.7788 0.9856 0.8701 0.9325
Dnn 0.9991 0.9968 0.7866 0.9807 0.8730 0.9556
Gmm 0.9528 0.9561 0.1737 0.7730 0.2837 0.1630
Isolation Forest 0.8967 0.9020 0.0660 0.5876 0.1187 0.1660
Kmeans 0.6438 0.9414 0.0268 0.1192 0.0438 0.0267
Logistic Regression 0.9905 0.9929 0.6350 0.8694 0.7339 0.8461
Pca Anomaly 0.5426 0.9405 0.0179 0.0795 0.0292 0.0263
Random Forest 0.9999 0.9993 0.9484 0.9958 0.9715 0.9942
Svm 0.9973 0.9908 0.5524 0.9705 0.7041 0.8559
Xgboost 0.9997 0.9971 0.7970 0.9952 0.8851 0.9773
Ensemble 0.9996 0.9977 0.8499 0.9651 0.9039 0.9562
"""

# Replace spaces in multi-word model names with underscores for correct parsing
# This is necessary because the data is space-separated, and some model names contain spaces.
data_fixed = data.replace('Decision Tree', 'Decision_Tree').replace('Isolation Forest', 'Isolation_Forest').replace('Logistic Regression', 'Logistic_Regression').replace('Pca Anomaly', 'Pca_Anomaly').replace('Random Forest', 'Random_Forest')

# Load data and replace underscores back to spaces for better labels
df = pd.read_csv(io.StringIO(data_fixed), sep='\s+')
df['Model'] = df['Model'].str.replace('_', ' ')

# --- 1. Bar Chart of AUC-ROC (Sorted) ---
df_auc = df.sort_values('AUC-ROC', ascending=False)
plt.figure(figsize=(12, 6))
plt.bar(df_auc['Model'], df_auc['AUC-ROC'], color='skyblue')
plt.ylabel('AUC-ROC Score')
plt.title('Model Performance: AUC-ROC Scores (Sorted)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0.5, 1.05)
plt.tight_layout()
plt.savefig('auc_roc_bar_chart.png')
plt.close()

# --- 2. Grouped Bar Chart for Key Metrics (Accuracy, Precision, Recall, F1) ---
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
df_metrics = df.set_index('Model')[metrics]
fig, ax = plt.subplots(figsize=(14, 7))
bar_width = 0.2
index = np.arange(len(df_metrics.index))
for i, metric in enumerate(metrics):
    ax.bar(index + i * bar_width, df_metrics[metric], bar_width, label=metric)
ax.set_xlabel('Model')
ax.set_ylabel('Score')
ax.set_title('Model Performance Comparison: Key Classification Metrics')
ax.set_xticks(index + bar_width * (len(metrics) - 1) / 2)
ax.set_xticklabels(df_metrics.index, rotation=45, ha='right')
ax.legend(loc='lower right', bbox_to_anchor=(1, 0.05))
ax.set_ylim(0.0, 1.05)
plt.tight_layout()
plt.savefig('grouped_metrics_bar_chart.png')
plt.close()

# --- 3. Scatter Plot of Recall vs. Precision (F1 Score Color-coded) ---
plt.figure(figsize=(8, 8))
scatter = plt.scatter(df['Recall'], df['Precision'], c=df['F1'], cmap='viridis', s=200, alpha=0.8)
for i, model in enumerate(df['Model']):
    plt.annotate(model, (df['Recall'][i] + 0.005, df['Precision'][i] + 0.005), fontsize=8, ha='left', va='bottom')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision vs. Recall Trade-off (F1 Score Color-coded)')
plt.xlim(0, 1.05)
plt.ylim(0, 1.05)
cbar = plt.colorbar(scatter)
cbar.set_label('F1 Score')
plt.tight_layout()
plt.savefig('recall_vs_precision_scatter_plot.png')
plt.close()