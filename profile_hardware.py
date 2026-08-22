import pandas as pd
import numpy as np
import os
import time
import joblib
import psutil
import tensorflow as tf
from ml_utils import add_gaussian_noise

# Path Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "processed_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

def measure_footprint(name, model_type, model_path, test_data):
    """Measures RAM, CPU, and Latency for a specific model."""
    print(f"Profiling {name}...")
    
    # Pre-loading process info
    process = psutil.Process(os.getpid())
    ram_before = process.memory_info().rss / (1024 * 1024) # MB
    
    # Load Model
    start_load = time.time()
    if model_type == 'keras':
        model = tf.keras.models.load_model(model_path)
    else:
        model = joblib.load(model_path)
    load_time = time.time() - start_load
    
    ram_after_load = process.memory_info().rss / (1024 * 1024) # MB
    
    # Inference Benchmarking
    X = test_data.drop(columns=['target']).head(1000) # Benchmark on 1000 samples
    
    # Warm up
    if model_type == 'keras':
        # Handle 3D for CNN/RNN if needed
        if name in ["CNN", "LSTM", "GRU"]:
            X_bench = X.values.reshape((X.shape[0], X.shape[1], 1))
        else:
            X_bench = X.values
        _ = model.predict(X_bench, verbose=0)
    else:
        _ = model.predict(X)
        
    # Real measurement
    start_inf = time.time()
    cpu_percent_start = process.cpu_percent(interval=None)
    
    if model_type == 'keras':
        _ = model.predict(X_bench, verbose=0)
    else:
        _ = model.predict(X)
        
    inf_time = time.time() - start_inf
    cpu_percent_end = process.cpu_percent(interval=None)
    
    latency_ms = (inf_time / 1000) * 1000 # per-sample ms
    
    return {
        "Model": name,
        "Load_Time_s": load_time,
        "RAM_Usage_MB": ram_after_load - ram_before,
        "Inference_Latency_ms": latency_ms,
        "CPU_Usage_Peak_%": cpu_percent_end,
        "Model_Size_MB": os.path.getsize(model_path) / (1024 * 1024)
    }

def main():
    if not os.path.exists(DATA_DIR):
        print("Data directory not found. Please run preprocessing first.")
        return
        
    test_df = pd.read_pickle(os.path.join(DATA_DIR, "test_data.pkl"))
    
    models_to_profile = [
        ("DNN", "keras", "dnn.keras"),
        ("CNN", "keras", "cnn.keras"),
        ("LSTM", "keras", "lstm.keras"),
        ("GRU", "keras", "gru.keras"),
        ("LightGBM", "sklearn", "lightgbm.pkl"),
        ("XGBoost", "sklearn", "xgboost.pkl"),
        ("Random Forest", "sklearn", "random_forest.pkl"),
        ("SVM", "sklearn", "svm.pkl"),
        ("Decision Tree", "sklearn", "decision_tree.pkl"),
        ("Autoencoder", "keras", "ae_model.keras"),
        ("Isolation Forest", "sklearn", "if_model.joblib")
    ]
    
    results = []
    for name, mtype, filename in models_to_profile:
        path = os.path.join(MODEL_DIR, filename)
        if os.path.exists(path):
            try:
                res = measure_footprint(name, mtype, path, test_df)
                results.append(res)
            except Exception as e:
                print(f"Error profiling {name}: {e}")
        else:
            print(f"Skipping {name}, file not found: {filename}")
            
    df_results = pd.DataFrame(results)
    output_path = os.path.join(OUTPUT_DIR, "hardware_footprint.csv")
    df_results.to_csv(output_path, index=False)
    print(f"\nProfiling complete. Results saved to {output_path}")
    print(df_results)

if __name__ == "__main__":
    main()
