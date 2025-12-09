import pandas as pd
import io
import matplotlib.pyplot as plt
import numpy as np

# === Your data ===
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

# --- Fix names for parsing ---
data_fixed = (data.replace('Decision Tree', 'Decision_Tree')
                  .replace('Isolation Forest', 'Isolation_Forest')
                  .replace('Logistic Regression', 'Logistic_Regression')
                  .replace('Pca Anomaly', 'Pca_Anomaly')
                  .replace('Random Forest', 'Random_Forest'))

df = pd.read_csv(io.StringIO(data_fixed), sep='\s+')
df['Model'] = df['Model'].str.replace('_', ' ')

# === Smart Abbreviation Mapping (for clean, readable labels) ===
abbrev = {
    'Autoencoder': 'Autoencoder',
    'Cnn': 'CNN',
    'Dbscan': 'DBSCAN',
    'Decision Tree': 'DT',
    'Dnn': 'DNN',
    'Gmm': 'GMM',
    'Isolation Forest': 'Isolation Forest',
    'Kmeans': 'KMeans',
    'Logistic Regression': 'LR',
    'Pca Anomaly': 'PCA Anomaly',
    'Random Forest': 'RF',
    'Svm': 'SVM',
    'Xgboost': 'XGBoost',
    'Ensemble': 'Ensemble'
}
df['Label'] = df['Model'].map(abbrev)

# === Group Models ===
classification_models = ['CNN', 'DT', 'DNN', 'LR', 'RF', 'SVM', 'XGBoost', 'Ensemble']
clustering_models = ['Autoencoder', 'DBSCAN', 'GMM', 'Isolation Forest', 'KMeans', 'PCA Anomaly']

df_class = df[df['Label'].isin(classification_models)].copy()
df_clus = df[df['Label'].isin(clustering_models)].copy()

# === Improved Plotting Function ===
def generate_bar_chart(df_group, title, filename):
    df_plot = df_group.set_index('Label')[['AUC-ROC', 'F1', 'Precision', 'Recall']].sort_values('AUC-ROC', ascending=False)
    
    fig, ax = plt.subplots(figsize=(14, 8))  # Bigger figure
    bar_width = 0.2
    index = np.arange(len(df_plot))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # AUC, F1, Prec, Rec
    metrics = ['AUC-ROC', 'F1', 'Precision', 'Recall']
    labels = ['AUC-ROC', 'F1 Score', 'Precision', 'Recall']

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        ax.bar(index + i * bar_width, df_plot[metric], bar_width,
               label=label, color=color, edgecolor='black', linewidth=0.7)

    ax.set_xlabel('Model', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_ylabel('Score', fontsize=16, fontweight='bold', labelpad=15)
    ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.set_xticks(index + bar_width * 1.5)
    ax.set_xticklabels(df_plot.index, rotation=45, ha='right', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
    ax.grid(axis='y', linestyle='--', alpha=0.7, linewidth=0.8)
    
    # Increase tick label size
    ax.tick_params(axis='both', labelsize=16)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

# === Generate High-Quality Plots ===
generate_bar_chart(
    df_class,
    'Supervised Classification Models Performance',
    'classification_model_performance.png'
)

generate_bar_chart(
    df_clus,
    'Unsupervised Anomaly Detection Models Performance',
    'clustering_model_performance.png'
)

print("\nBoth plots generated successfully with large, readable fonts and smart abbreviations!")