import pandas as pd
import numpy as np
import os
import joblib
import tensorflow as tf
from ml_utils import evaluate_model
import json

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_EXT_DIR = os.path.join(BASE_DIR, "processed_external")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "results", "cross_validation")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

class DLModelWrapper:
    def __init__(self, model, name):
        self.model = model
        self.name = name
        self.is_3d = name in ["CNN", "LSTM", "GRU"]

    def predict(self, X):
        if self.is_3d:
            X = X.values.reshape((X.shape[0], X.shape[1], 1))
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)

def run_cross_validation():
    ext_files = [f for f in os.listdir(PROCESSED_EXT_DIR) if f.endswith("_aligned.pkl")]
    
    # Load Models
    models = {}
    print("Loading models...")
    # ML Models
    for m_name in ["lightgbm", "xgboost", "random_forest", "svm", "decision_tree"]:
        path = os.path.join(MODEL_DIR, f"{m_name}.pkl")
        if os.path.exists(path):
            models[m_name] = joblib.load(path)
            
    # DL Models
    for m_name in ["dnn", "cnn", "lstm", "gru"]:
        path = os.path.join(MODEL_DIR, f"{m_name}.keras")
        if os.path.exists(path):
            keras_m = tf.keras.models.load_model(path)
            models[m_name] = DLModelWrapper(keras_m, m_name.upper())

    all_results = []
    
    for ext_file in ext_files:
        dataset_name = ext_file.split("_")[0].upper()
        print(f"\nEvaluating on {dataset_name}...")
        
        df = pd.read_pickle(os.path.join(PROCESSED_EXT_DIR, ext_file))
        X = df.drop(columns=['target'])
        y = df['target']
        
        dataset_results = []
        for name, model in models.items():
            print(f"  Testing {name}...")
            metrics = evaluate_model(model, X, y, name, quiet=True)
            metrics['Dataset'] = dataset_name
            dataset_results.append(metrics)
            
        all_results.extend(dataset_results)
        
        # Save per-dataset results for backup
        pd.DataFrame(dataset_results).to_csv(os.path.join(OUTPUT_DIR, f"{dataset_name.lower()}_results.csv"), index=False)

    # Save Aggregate Results
    pd.DataFrame(all_results).to_csv(os.path.join(OUTPUT_DIR, "aggregate_cross_val_results.csv"), index=False)
    print(f"\nCross-Validation complete. Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_cross_validation()
