import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def plot_confusion_matrix(y_true, y_pred, model_name, output_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"cm_{model_name.lower().replace(' ', '_')}.png"))
    plt.close()

def add_gaussian_noise(X, noise_level=0.05):
    """
    Adds Gaussian noise to the dataset features to simulate wireless interference.
    noise_level: Standard deviation of the noise relative to the feature's std (which is 1.0 after scaling).
    """
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return X_noisy

def evaluate_model(model, X_test, y_test, model_name, output_dir=None, quiet=False):
    # Inference Latency
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    inference_time = end_time - start_time
    latency_per_sample_ms = (inference_time / len(X_test)) * 1000
    
    # Classification Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    if not quiet:
        print(f"[{model_name}] Accuracy: {accuracy:.4f}, Latency: {latency_per_sample_ms:.4f} ms/sample")
    
    # Save Confusion Matrix
    if output_dir:
        plot_confusion_matrix(y_test, y_pred, model_name, output_dir)
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Latency_ms": latency_per_sample_ms
    }

def robustness_test(model, X_test, y_test, model_name):
    # Standard robustness test with varying noise levels
    noise_levels = [0.01, 0.05, 0.1, 0.2]
    results = []
    
    for lvl in noise_levels:
        X_noisy = add_gaussian_noise(X_test, noise_level=lvl)
        
        # Get full metrics for robustness
        metrics = evaluate_model(model, X_noisy, y_test, f"{model_name}_Noise_{lvl}", quiet=True)
        results.append(metrics)
        
    return noise_levels, results
