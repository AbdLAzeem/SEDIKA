import pandas as pd
import numpy as np
import os
import time
import pickle
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from paths import PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, PLOT_DIR, ensure_dirs

ensure_dirs(OUTPUT_DIR, MODEL_DIR, PLOT_DIR)

_SEED = int(os.environ.get("SEDIKA_SEED", 42))
tf.random.set_seed(_SEED)
np.random.seed(_SEED)

def load_data():
    # Anomaly model trains on the un-resampled pool to keep the benign manifold authentic.
    training_set = pd.read_pickle(os.path.join(PROCESSED_DIR, "train_data.pkl"))
    test_set = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))
    from artifacts import load_artifact
    le = load_artifact(os.path.join(PROCESSED_DIR, "label_encoder.joblib"))
    classes = le.classes_
    print(f"Classes: {classes}")
    
    # Heuristic: usually 'Thing_Speak' or 'MQTT_Publish' represents normal operation in this context
    # Let's verify. If ambiguous, we consider the majority class as Normal or explicit labels.
    # Actually, often 'Thing_Speak' is the authorized behavior.
    # Let's find index for 'Thing_Speak'
    normal_indices = [i for i, c in enumerate(classes) if 'Thing_Speak' in c]
    
    if not normal_indices:
        print("Warning: 'Thing_Speak' class not found. Trying 'MQTT_Publish' or similar.")
        normal_indices = [i for i, c in enumerate(classes) if 'MQTT' in c]
        
    if not normal_indices:
        print("Warning: No clear Normal class found. Using class 0 as Normal.")
        normal_label = 0
    else:
        normal_label = normal_indices[0] # Take the first match
        
    print(f"Treating class '{classes[normal_label]}' (ID: {normal_label}) as NORMAL.")
    
    y_train = training_set['target']
    X_train = training_set.drop(columns=['target'])
    
    y_test = test_set['target']
    X_test = test_set.drop(columns=['target'])
    
    # Create Binary Targets for Anomaly Detection (0=Normal, 1=Anomaly)
    # Note: Scikit-learn IF uses 1=Normal, -1=Anomaly. We will adjust.
    # For Autoencoder, we train on Normal only.
    
    # Filtering Normal Train Data
    X_train_normal = X_train[y_train == normal_label]
    print(f"Normal Training Samples: {len(X_train_normal)}")
    
    # For Test, we keep all, but create binary labels (0=Normal, 1=Anomaly)
    y_test_binary = (y_test != normal_label).astype(int)
    print(f"Test Anomaly Ratio: {y_test_binary.mean():.2f}")
    
    return X_train_normal, X_test, y_test_binary

def train_isolation_forest(X_train, X_test, y_test):
    print("\nTraining Isolation Forest...")
    clf = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    
    t0 = time.time()
    clf.fit(X_train)
    train_time = time.time() - t0
    
    # Predict (returns 1 for inliers, -1 for outliers)
    y_pred_raw = clf.predict(X_test)
    y_scores = -clf.score_samples(X_test) # Higher score = more anomalous
    
    # Convert to 0=Normal, 1=Anomaly
    y_pred = np.where(y_pred_raw == -1, 1, 0)
    
    auc_score = roc_auc_score(y_test, y_scores)
    print(f"[Isolation Forest] AUROC: {auc_score:.4f}, Train Time: {train_time:.2f}s")
    
    # Plot ROC
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Isolation Forest (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve - Anomaly Detection')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "roc_isolation_forest.png"))
    plt.close()
    
    return auc_score

def train_autoencoder(X_train, X_test, y_test):
    print("\nTraining Autoencoder...")
    input_dim = X_train.shape[1]
    
    # Architecture
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation="relu")(input_layer)
    encoder = Dense(8, activation="relu")(encoder)
    decoder = Dense(16, activation="relu")(encoder)
    decoder = Dense(input_dim, activation="linear")(decoder) # Reconstruction
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    t0 = time.time()
    autoencoder.fit(X_train, X_train,
                    epochs=20,
                    batch_size=64,
                    shuffle=True,
                    verbose=1)
    train_time = time.time() - t0
    
    # Predict (Reconstruction)
    reconstructions = autoencoder.predict(X_test)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
    
    auc_score = roc_auc_score(y_test, mse)
    print(f"[Autoencoder] AUROC: {auc_score:.4f}, Train Time: {train_time:.2f}s")
    
    # Plot ROC
    fpr, tpr, _ = roc_curve(y_test, mse)
    plt.figure()
    plt.plot(fpr, tpr, color='orange', label=f'Autoencoder (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve - Anomaly Detection')
    plt.legend()
    plt.savefig(os.path.join(PLOT_DIR, "roc_autoencoder.png"))
    plt.close()
    
    return auc_score

def add_gaussian_noise(X, noise_level=0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise

def run_anomaly_detection():
    X_train_normal, X_test, y_test_binary = load_data()
    
    # 1. Isolation Forest
    print("\nTraining Isolation Forest...")
    clf = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
    t0 = time.time()
    clf.fit(X_train_normal)
    train_time_if = time.time() - t0
    
    # Save IF Model
    if_model_path = os.path.join(MODEL_DIR, "if_model.joblib")
    joblib.dump(clf, if_model_path)
    
    # Evaluate IF
    y_scores_if_clear = -clf.score_samples(X_test)
    if_auc_clear = roc_auc_score(y_test_binary, y_scores_if_clear)
    
    X_test_noisy = add_gaussian_noise(X_test, noise_level=0.1)
    y_scores_if_noisy = -clf.score_samples(X_test_noisy)
    if_auc_noisy = roc_auc_score(y_test_binary, y_scores_if_noisy)
    
    # 2. Autoencoder
    print("\nTraining Autoencoder...")
    input_dim = X_train_normal.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation="relu")(input_layer)
    encoder = Dense(8, activation="relu")(encoder)
    decoder = Dense(16, activation="relu")(encoder)
    decoder = Dense(input_dim, activation="linear")(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    t0 = time.time()
    autoencoder.fit(X_train_normal, X_train_normal,
                    epochs=20,
                    batch_size=64,
                    shuffle=True,
                    verbose=0)
    train_time_ae = time.time() - t0
    
    autoencoder.save(os.path.join(MODEL_DIR, "ae_model.keras"))

    # Evaluate AE
    def get_ae_scores(model, X):
        reconstructions = model.predict(X, verbose=0)
        return np.mean(np.power(X - reconstructions, 2), axis=1)

    mse_clear = get_ae_scores(autoencoder, X_test)
    ae_auc_clear = roc_auc_score(y_test_binary, mse_clear)
    
    mse_noisy = get_ae_scores(autoencoder, X_test_noisy)
    ae_auc_noisy = roc_auc_score(y_test_binary, mse_noisy)
    
    # Save Results
    results = pd.DataFrame({
        "Model": ["Isolation Forest", "Autoencoder"],
        "AUROC_Clear": [if_auc_clear, ae_auc_clear],
        "AUROC_Noisy_0.1": [if_auc_noisy, ae_auc_noisy],
        "Train_Time_s": [train_time_if, train_time_ae]
    })
    results.to_csv(os.path.join(OUTPUT_DIR, "anomaly_detection_results.csv"), index=False)
    print("\nAnomaly Detection Complete.")
    print("Identification: This evaluation covers BOTH Clear and Noisy (Level 0.1) scenarios.")
    print(results)

if __name__ == "__main__":
    run_anomaly_detection()
