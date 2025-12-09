import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.ensemble import RandomForestClassifier
import datetime
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, RocCurveDisplay
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from tensorflow.keras.layers import LSTM, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from imblearn.over_sampling import SMOTE

# Load the dataset
def load_data(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    print(f"Initial shape: {df.shape}")

    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = pd.to_numeric(df[column], errors='coerce')
    print(f"After numeric conversion, shape: {df.shape}")

    initial_rows = len(df)
    df = df.dropna(subset=['is_fraudulent'])
    dropped_rows = initial_rows - len(df)
    print(f"Dropped {dropped_rows} rows due to NaN in is_fraudulent, new shape: {df.shape}")

    df = df.replace([np.inf, -np.inf], np.nan)
    print(f"After replacing inf with NaN, shape: {df.shape}")

    df = df.fillna(0)
    print(f"After filling NaN with 0, shape: {df.shape}")

    df = df.drop(['fyear', 'gvkey'], axis=1, errors='ignore')
    print(f"Final shape after dropping fyear and gvkey: {df.shape}")

    return df


# Function to create model-specific folder
def create_model_folder(model_name):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"MyModels/{model_name}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


# Function to save results, plots, confusion matrix, and model
def save_model_artifacts(folder, model_name, metrics, hyperparameters, model, scaler, y_test, y_pred, y_pred_prob):
    # Save metrics and hyperparameters
    with open(f"{folder}/metrics.txt", 'w') as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Timestamp: {os.path.basename(folder).split('_', 1)[1]}\n")
        f.write("Performance Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"  {metric}: {value:.4f}\n")
        f.write("Hyperparameters:\n")
        for param, value in hyperparameters.items():
            f.write(f"  {param}: {value}\n")
        f.write("-" * 50 + "\n")

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"{folder}/confusion_matrix.png", bbox_inches='tight')
    plt.close()

    # Save ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    roc_display.plot()
    plt.title(f"ROC Curve - {model_name}")
    plt.savefig(f"{folder}/roc_curve.png", bbox_inches='tight')
    plt.close()

    # Save model and scaler
    joblib.dump(model, f"{folder}/model.pkl")
    joblib.dump(scaler, f"{folder}/scaler.pkl")

    print(f"All artifacts saved in: {folder}")


# Function for XGBoost model
def run_xgboost(df):
    X = df.drop('is_fraudulent', axis=1)
    y = df['is_fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_prob)
    }

    hyperparameters = {
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 100
    }

    folder = create_model_folder('xgboost')
    save_model_artifacts(folder, 'XGBoost', metrics, hyperparameters, model, scaler, y_test, y_pred, y_pred_prob)

    print(f"XGBoost Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model


# Function for Isolation Forest
def run_isolation_forest(df):
    X = df.drop('is_fraudulent', axis=1)
    y_true = df['is_fraudulent']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42
    )
    model.fit(X_scaled)

    y_pred = model.predict(X_scaled)
    y_pred_binary = [1 if x == -1 else 0 for x in y_pred]
    y_scores = -model.score_samples(X_scaled)  # Higher = more anomalous

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary),
        'recall': recall_score(y_true, y_pred_binary),
        'f1_score': f1_score(y_true, y_pred_binary),
        'auc_roc': roc_auc_score(y_true, y_scores)
    }

    hyperparameters = {
        'n_estimators': 100,
        'contamination': 0.1
    }

    folder = create_model_folder('isolation_forest')
    save_model_artifacts(folder, 'Isolation Forest', metrics, hyperparameters, model, scaler, y_true, y_pred_binary, y_scores)

    print(f"Isolation Forest Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model


# Function for DNN model
def run_dnn(df):
    X = df.drop('is_fraudulent', axis=1)
    y = df['is_fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    y_pred_prob = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_prob)
    }

    hyperparameters = {
        'layers': [64, 32, 1],
        'activation': 'relu',
        'learning_rate': 0.001,
        'epochs': 50,
        'batch_size': 32
    }

    folder = create_model_folder('dnn')
    save_model_artifacts(folder, 'DNN', metrics, hyperparameters, model, scaler, y_test, y_pred, y_pred_prob)

    print(f"DNN Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model


# Function for Random Forest
def run_random_forest(df):
    X = df.drop('is_fraudulent', axis=1)
    y = df['is_fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        bootstrap=True,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_prob)
    }

    hyperparameters = {
        'n_estimators': 200,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'bootstrap': True
    }

    folder = create_model_folder('random_forest')
    save_model_artifacts(folder, 'Random Forest', metrics, hyperparameters, model, scaler, y_test, y_pred, y_pred_prob)

    print(f"Random Forest Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model

def run_dbscan(df):
    X = df.drop('is_fraudulent', axis=1)
    y_true = df['is_fraudulent']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = DBSCAN(eps=0.5, min_samples=5)
    clusters = model.fit_predict(X_scaled)
    y_pred = [1 if label == -1 else 0 for label in clusters]  # -1 = anomaly

    # Use cluster scores as anomaly probability (distance to nearest cluster)
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
    distances, _ = nbrs.kneighbors(X_scaled)
    y_scores = distances[:, 0]  # Distance to nearest neighbor

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_scores)
    }

    hyperparameters = {'eps': 0.5, 'min_samples': 5}

    folder = create_model_folder('dbscan')
    save_model_artifacts(folder, 'DBSCAN', metrics, hyperparameters, model, scaler, y_true, y_pred, y_scores)

    print(f"DBSCAN Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model

def run_autoencoder(df):
    X = df.drop('is_fraudulent', axis=1)
    y_true = df['is_fraudulent']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]
    encoding_dim = 32

    # Build Autoencoder
    input_layer = tf.keras.Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(encoding_dim, activation='relu')(encoded)
    decoded = Dense(64, activation='relu')(encoded)
    decoded = Dense(input_dim, activation='linear')(decoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

    # Reconstruction error
    reconstructed = autoencoder.predict(X_scaled)
    mse = np.mean(np.power(X_scaled - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, 95)  # Top 5% as anomalies
    y_pred = (mse > threshold).astype(int)
    y_scores = mse

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_scores)
    }

    hyperparameters = {'encoding_dim': encoding_dim, 'epochs': 50, 'batch_size': 32}

    folder = create_model_folder('autoencoder')
    # Save Keras model as .h5
    autoencoder.save(f"{folder}/model.h5")
    joblib.dump(scaler, f"{folder}/scaler.pkl")
    save_model_artifacts(folder, 'Autoencoder', metrics, hyperparameters, autoencoder, scaler, y_true, y_pred, y_scores)

    print(f"Autoencoder Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return autoencoder

def run_logistic_regression(df):
    X = df.drop('is_fraudulent', axis=1)
    y = df['is_fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_prob)
    }

    hyperparameters = {'max_iter': 1000}

    folder = create_model_folder('logistic_regression')
    save_model_artifacts(folder, 'Logistic Regression', metrics, hyperparameters, model, scaler, y_test, y_pred, y_pred_prob)

    print(f"Logistic Regression - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model

def run_svm(df):
    X = df.drop('is_fraudulent', axis=1)
    y = df['is_fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVC(kernel='rbf', probability=True, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_prob)
    }

    hyperparameters = {'kernel': 'rbf'}

    folder = create_model_folder('svm')
    save_model_artifacts(folder, 'SVM', metrics, hyperparameters, model, scaler, y_test, y_pred, y_pred_prob)

    print(f"SVM Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model

def run_kmeans(df):
    X = df.drop('is_fraudulent', axis=1)
    y_true = df['is_fraudulent']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KMeans(n_clusters=5, random_state=42)
    model.fit(X_scaled)
    distances = model.transform(X_scaled)
    dist_to_cluster = np.min(distances, axis=1)
    threshold = np.percentile(dist_to_cluster, 95)
    y_pred = (dist_to_cluster > threshold).astype(int)
    y_scores = dist_to_cluster

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_scores)
    }

    hyperparameters = {'n_clusters': 5}

    folder = create_model_folder('kmeans')
    save_model_artifacts(folder, 'K-Means', metrics, hyperparameters, model, scaler, y_true, y_pred, y_scores)

    print(f"K-Means Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model

def run_gmm(df):
    X = df.drop('is_fraudulent', axis=1)
    y_true = df['is_fraudulent']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
    model.fit(X_scaled)
    scores = model.score_samples(X_scaled)
    threshold = np.percentile(scores, 5)  # Bottom 5% as anomalies
    y_pred = (scores < threshold).astype(int)
    y_scores = -scores  # Invert: higher = more anomalous

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_scores)
    }

    hyperparameters = {'n_components': 3, 'covariance_type': 'full'}

    folder = create_model_folder('gmm')
    save_model_artifacts(folder, 'GMM', metrics, hyperparameters, model, scaler, y_true, y_pred, y_scores)

    print(f"GMM Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model

def run_lstm(df):
    X = df.drop('is_fraudulent', axis=1).values
    y = df['is_fraudulent'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape for LSTM: (samples, 10, features//10)
    n_features = X_scaled.shape[1]
    timesteps = 10
    if n_features < timesteps:
        timesteps = 1
        X_reshaped = X_scaled.reshape(-1, 1, n_features)
    else:
        X_padded = np.hstack([X_scaled, np.zeros((X_scaled.shape[0], timesteps - (n_features % timesteps)))])
        X_reshaped = X_padded.reshape(-1, timesteps, -1)

    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(LSTM(50, input_shape=(timesteps, X_reshaped.shape[2]), return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_prob)
    }

    hyperparameters = {'lstm_units': 50, 'epochs': 20, 'timesteps': timesteps}

    folder = create_model_folder('lstm')
    model.save(f"{folder}/model.h5")
    joblib.dump(scaler, f"{folder}/scaler.pkl")
    save_model_artifacts(folder, 'LSTM', metrics, hyperparameters, model, scaler, y_test, y_pred, y_pred_prob)

    print(f"LSTM Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model

def run_decision_tree(df):
    X = df.drop('is_fraudulent', axis=1)
    y = df['is_fraudulent']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = DecisionTreeClassifier(max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_prob)
    }

    hyperparameters = {'max_depth': 10}

    folder = create_model_folder('decision_tree')
    save_model_artifacts(folder, 'Decision Tree', metrics, hyperparameters, model, scaler, y_test, y_pred, y_pred_prob)

    print(f"Decision Tree - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model

def run_pca_anomaly(df):
    X = df.drop('is_fraudulent', axis=1)
    y_true = df['is_fraudulent']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95)  # Keep 95% variance
    X_pca = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_pca)
    mse = np.mean(np.power(X_scaled - X_reconstructed, 2), axis=1)

    threshold = np.percentile(mse, 95)
    y_pred = (mse > threshold).astype(int)
    y_scores = mse

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_scores)
    }

    hyperparameters = {'n_components': '95% variance'}

    folder = create_model_folder('pca_anomaly')
    joblib.dump(pca, f"{folder}/model.pkl")
    joblib.dump(scaler, f"{folder}/scaler.pkl")
    save_model_artifacts(folder, 'PCA Anomaly', metrics, hyperparameters, pca, scaler, y_true, y_pred, y_scores)

    print(f"PCA Anomaly - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return pca

def run_cnn(df):
    X = df.drop('is_fraudulent', axis=1).values
    y = df['is_fraudulent'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reshape: (samples, features, 1)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_reshaped.shape[1], 1)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(32, kernel_size=3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)

    y_pred_prob = model.predict(X_test).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'auc_roc': roc_auc_score(y_test, y_pred_prob)
    }

    hyperparameters = {'conv_layers': [64, 32], 'epochs': 20}

    folder = create_model_folder('cnn')
    model.save(f"{folder}/model.h5")
    joblib.dump(scaler, f"{folder}/scaler.pkl")
    save_model_artifacts(folder, 'CNN', metrics, hyperparameters, model, scaler, y_test, y_pred, y_pred_prob)

    print(f"CNN Results - Accuracy: {metrics['accuracy']:.4f}, AUC-ROC: {metrics['auc_roc']:.4f}")
    return model

def run_lightgbm(df):
    X = df.drop('is_fraudulent', axis=1)
    y = df['is_fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    # NEW CORRECT WAY (LightGBM >= 4.0)
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        callbacks=[early_stopping(stopping_rounds=50, verbose=True)]
    )

    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_prob)
    }

    hyperparameters = {
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'early_stopping_rounds': 50
    }

    folder = create_model_folder('lightgbm')
    save_model_artifacts(
        folder, 'LightGBM', metrics, hyperparameters,
        model, scaler, y_test, y_pred, y_pred_prob
    )
    print(f"LightGBM → AUC-ROC: {metrics['auc_roc']:.4f} | Best iteration: {model.best_iteration_}")
    return model

def run_catboost(df):
    X = df.drop('is_fraudulent', axis=1)
    y = df['is_fraudulent']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3,
        border_count=128,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=100,
        eval_metric='AUC'
    )
    model.fit(X_train_scaled, y_train, eval_set=(X_test_scaled, y_test), use_best_model=True)

    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_test, y_pred_prob)
    }

    hyperparameters = {
        'iterations': 800,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3
    }

    folder = create_model_folder('catboost')
    save_model_artifacts(
        folder, 'CatBoost', metrics, hyperparameters,
        model, scaler, y_test, y_pred, y_pred_prob
    )
    print(f"CatBoost → AUC-ROC: {metrics['auc_roc']:.4f}, Recall: {metrics['recall']:.4f}")
    return model

def run_oneclass_svm(df):
    # Use only normal (non-fraud) samples for training (true unsupervised)
    normal_df = df[df['is_fraudulent'] == 0]
    X_normal = normal_df.drop('is_fraudulent', axis=1)

    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)

    model = OneClassSVM(
        kernel='rbf',
        gamma='scale',
        nu=0.05,           # Expected outlier ratio (tune this!)
        verbose=False
    )
    model.fit(X_normal_scaled)

    # Test on full dataset
    X_full = df.drop('is_fraudulent', axis=1)
    X_full_scaled = scaler.transform(X_full)
    y_true = df['is_fraudulent'].values

    # OneClassSVM: +1 = normal, -1 = anomaly
    scores = model.decision_function(X_full_scaled)
    pred_raw = model.predict(X_full_scaled)
    y_pred = np.where(pred_raw == -1, 1, 0)  # -1 → fraud

    # Convert decision scores to probability-like (higher = more anomalous)
    prob = -scores
    prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, prob)
    }

    hyperparameters = {
        'kernel': 'rbf',
        'gamma': 'scale',
        'nu': 0.05
    }

    folder = create_model_folder('oneclass_svm')
    save_model_artifacts(
        folder, 'One-Class SVM', metrics, hyperparameters,
        model, scaler, y_true, y_pred, prob
    )
    print(f"One-Class SVM → AUC-ROC: {metrics['auc_roc']:.4f}, Detected: {y_pred.sum()}/{len(y_pred)}")
    return model

def run_lof(df):
    # Train on normal data only
    normal_df = df[df['is_fraudulent'] == 0]
    X_normal = normal_df.drop('is_fraudulent', axis=1)

    scaler = StandardScaler()
    X_normal_scaled = scaler.fit_transform(X_normal)

    model = LocalOutlierFactor(
        n_neighbors=20,
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        p=2,
        contamination=0.05,   # Expected fraud rate
        novelty=True           # Important: allows prediction on new data
    )
    model.fit(X_normal_scaled)

    X_full = df.drop('is_fraudulent', axis=1)
    X_full_scaled = scaler.transform(X_full)
    y_true = df['is_fraudulent'].values

    # Negative scores = more anomalous
    scores = model.decision_function(X_full_scaled)
    pred_raw = model.predict(X_full_scaled)
    y_pred = np.where(pred_raw == -1, 1, 0)

    prob = -scores
    prob = (prob - prob.min()) / (prob.max() - prob.min() + 1e-8)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, prob)
    }

    hyperparameters = {
        'n_neighbors': 20,
        'contamination': 0.05,
        'novelty': True
    }

    folder = create_model_folder('lof')
    save_model_artifacts(
        folder, 'Local Outlier Factor', metrics, hyperparameters,
        model, scaler, y_true, y_pred, prob
    )
    print(f"LOF → AUC-ROC: {metrics['auc_roc']:.4f}, Outliers flagged: {y_pred.sum()}")
    return model

# Main function


# --------------------------------------------------------------
# Helper: Prepare SMOTE-balanced data (only for supervised models)
# --------------------------------------------------------------
def prepare_smote_data(df, target='is_fraudulent', sampling_strategy=0.5, random_state=42):
    """
    Returns SMOTE-balanced train set + original (imbalanced) test set
    Only for supervised models!
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    print(f"\nBefore SMOTE → Fraud rate in train: {y_train.mean():.5f}")

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE  → Fraud rate in train: {y_train_smote.mean():.5f} "
          f"({y_train_smote.sum()} fraud / {len(y_train_smote)} total)")

    # Reconstruct DataFrames so your existing functions don't break
    train_df = pd.DataFrame(X_train_smote, columns=X.columns)
    train_df[target] = y_train_smote

    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df[target] = y_test

    return train_df, test_df


# ====================================================================
# YOUR FULL MAIN FUNCTION — COPY & PASTE THIS ENTIRE THING
# ====================================================================
def main():
    # Create folder for saved models
    os.makedirs("MyModels", exist_ok=True)

    # Load your dataset
    print("Loading dataset...")
    df = load_data('../Dataset/enhanced_financial_data.csv')
    print(f"Dataset loaded → {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Original fraud rate: {df['is_fraudulent'].mean()*100:.5f}%")

    # ========================== FLAGS ==========================
    run_xgboost_flag            = False
    run_isolation_forest_flag   = True
    run_dnn_flag                = False
    run_random_forest_flag      = False
    run_dbscan_flag             = True
    run_autoencoder_flag        = True
    run_logistic_flag           = False
    run_svm_flag                = False
    run_kmeans_flag             = True
    run_gmm_flag                = True
    run_lstm_flag               = False
    run_decision_tree_flag      = False
    run_pca_anomaly_flag        = False
    run_cnn_flag                = False

    # ←←← YOUR ACTIVE MODELS ↓↓↓
    run_lightgbm_flag           = False
    run_catboost_flag           = False
    run_oneclass_svm_flag       = False
    run_lof_flag                = False
    # ↑↑↑ Change these as needed

    # ========================== PREPARE SMOTE DATA FOR SUPERVISED MODELS ==========================
    supervised_models_will_run = (
        run_lightgbm_flag or run_catboost_flag or run_xgboost_flag or
        run_random_forest_flag or run_decision_tree_flag or run_logistic_flag or
        run_svm_flag or run_dnn_flag or run_lstm_flag or run_cnn_flag
    )

    if supervised_models_will_run:
        print("\n" + "="*70)
        print("PREPARING SMOTE-BALANCED DATASET FOR SUPERVISED MODELS")
        print("="*70)

        X = df.drop(columns=['is_fraudulent'])
        y = df['is_fraudulent']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Train set before SMOTE → {len(y_train):,} samples | Fraud rate: {y_train.mean():.6f}")

        # Apply SMOTE — change sampling_strategy as needed:
        # 1.0 = 50/50 balance | 0.5 = 1 fraud : 2 normal | 0.3 = 1:3 etc.
        smote = SMOTE(sampling_strategy=0.5, random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        print(f"Train set after SMOTE  → {len(y_train_resampled):,} samples | "
              f"Fraud rate: {y_train_resampled.mean():.4f} | "
              f"Added {y_train_resampled.sum() - y_train.sum():,} synthetic fraud cases")

        # Rebuild DataFrames
        train_res_df = X_train_resampled.copy()
        train_res_df['is_fraudulent'] = y_train_resampled

        test_res_df = X_test.copy()
        test_res_df['is_fraudulent'] = y_test

        # Final DataFrame passed to all supervised models (train + test combined)
        supervised_df = pd.concat([train_res_df, test_res_df], ignore_index=True)
        print(f"Final supervised dataset shape: {supervised_df.shape}")
        print(f"Final fraud rate (for supervised models): {supervised_df['is_fraudulent'].mean():.4f}\n")

    else:
        supervised_df = df.copy()  # fallback (not used)

    # ========================== RUN SUPERVISED MODELS (with SMOTE) ==========================
    if run_lightgbm_flag:
        print("\n=== Training LightGBM (with SMOTE) ===")
        run_lightgbm(supervised_df.copy())

    if run_catboost_flag:
        print("\n=== Training CatBoost (with SMOTE) ===")
        run_catboost(supervised_df.copy())

    if run_xgboost_flag:
        print("\n=== Training XGBoost (with SMOTE) ===")
        run_xgboost(supervised_df.copy())

    if run_random_forest_flag:
        print("\n=== Training Random Forest (with SMOTE) ===")
        run_random_forest(supervised_df.copy())

    if run_decision_tree_flag:
        print("\n=== Training Decision Tree (with SMOTE) ===")
        run_decision_tree(supervised_df.copy())

    if run_logistic_flag:
        print("\n=== Training Logistic Regression (with SMOTE) ===")
        run_logistic_regression(supervised_df.copy())

    if run_svm_flag:
        print("\n=== Training SVM (with SMOTE) ===")
        run_svm(supervised_df.copy())

    if run_dnn_flag:
        print("\n=== Training DNN (with SMOTE) ===")
        run_dnn(supervised_df.copy())

    if run_lstm_flag:
        print("\n=== Training LSTM (with SMOTE) ===")
        run_lstm(supervised_df.copy())

    if run_cnn_flag:
        print("\n=== Training CNN (with SMOTE) ===")
        run_cnn(supervised_df.copy())

    # ========================== RUN UNSUPERVISED MODELS (NO SMOTE — original data only) ==========================
    print("\n" + "="*70)
    print("RUNNING UNSUPERVISED MODELS ON ORIGINAL IMBALANCED DATA")
    print("="*70)

    if run_oneclass_svm_flag:
        print("\n=== Training One-Class SVM (unsupervised - NO SMOTE) ===")
        run_oneclass_svm(df.copy())

    if run_lof_flag:
        print("\n=== Training Local Outlier Factor (LOF - NO SMOTE) ===")
        run_lof(df.copy())

    if run_isolation_forest_flag:
        print("\n=== Training Isolation Forest (NO SMOTE) ===")
        run_isolation_forest(df.copy())

    if run_dbscan_flag:
        print("\n=== Training DBSCAN (NO SMOTE) ===")
        run_dbscan(df.copy())

    if run_kmeans_flag:
        print("\n=== Training K-Means (NO SMOTE) ===")
        run_kmeans(df.copy())

    if run_gmm_flag:
        print("\n=== Training GMM (NO SMOTE) ===")
        run_gmm(df.copy())

    if run_autoencoder_flag:
        print("\n=== Training Autoencoder (NO SMOTE) ===")
        run_autoencoder(df.copy())

    if run_pca_anomaly_flag:
        print("\n=== Training PCA Anomaly (NO SMOTE) ===")
        run_pca_anomaly(df.copy())

    # ========================== DONE ==========================
    print("\n" + "="*70)
    print("ALL TRAINING COMPLETED!")
    print("Check the 'MyModels' folder for saved models and results.")
    print("="*70)


# ====================================================================
# Run main when script is executed
# ====================================================================
if __name__ == "__main__":
    main()