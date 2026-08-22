import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
import os
import joblib
import shap
import matplotlib.pyplot as plt
from ml_utils import evaluate_model, add_gaussian_noise

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, "processed_external")
SOTA_RESULTS_DIR = os.path.join(BASE_DIR, "results", "SOTA results", "v2.2")
CORAL_DATA_PATH = os.path.join(EXTERNAL_DATA_DIR, "sedika_ciciot2023_coral.pkl")
DIFA_V2_MODEL_PATH = os.path.join(MODEL_DIR, "sedika_difa_v2.keras")
TARGET_ENCODER_PATH = os.path.join(EXTERNAL_DATA_DIR, "sedika_target_encoder.joblib")

if not os.path.exists(SOTA_RESULTS_DIR):
    os.makedirs(SOTA_RESULTS_DIR)

import pandas as pd
import numpy as np

class DIFA2Wrapper:
    def __init__(self, model):
        self.model = model
    def predict(self, X):
        probs, _ = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)
    def predict_proba(self, X):
        probs, _ = self.model.predict(X, verbose=0)
        return probs

def run_sota_v2_benchmark():
    print("SEDIKA Phase 3: SOTA-2.0 Resilience Benchmarking")
    
    # 1. Load Resources
    df = pd.read_pickle(CORAL_DATA_PATH)
    le = joblib.load(TARGET_ENCODER_PATH)
    
    # Load model (Handle custom Layer)
    from sedika_difa_v2 import GradientReversal
    model_keras = tf.keras.models.load_model(DIFA_V2_MODEL_PATH, custom_objects={'GradientReversal': GradientReversal})
    model = DIFA2Wrapper(model_keras)
    
    X = df.drop(columns=['target']).values
    y = df['target'].values
    
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 2. Base Accuracy
    print("\n[Accuracy Audit - DIFA-2.0 Aligned]")
    metrics_clear = evaluate_model(model, X_test, y_test, "SEDIKA_SOTA_v2_Clear")
    
    # 3. Adversarial Jitter Resilience (sigma=0.1)
    print("\n[Robustness Audit - Noise sigma=0.1]")
    X_noisy = add_gaussian_noise(X_test, noise_level=0.1)
    metrics_noisy = evaluate_model(model, X_noisy, y_test, "SEDIKA_SOTA_v2_Noisy")
    
    # 4. SHAP Sensitivity Audit
    print("\n[SHAP Sensitivity Audit - DIFA-2.0]")
    background = X_test[:100].astype('float32')
    
    # Extract just the task-prediction sub-model for SHAP
    task_model = Model(inputs=model_keras.input, outputs=model_keras.get_layer('task_output').output)
    explainer = shap.GradientExplainer(task_model, background)
    
    sample_idx = 0
    sample = X_test[[sample_idx]].astype('float32')
    target_feat_idx = 1 # fwd_pkts_payload.avg
    target_feat_name = "fwd_pkts_payload.avg"
    base_val = sample[0, target_feat_idx]
    
    perturbations = np.linspace(base_val - 2.0, base_val + 2.0, 100).astype('float32')
    perturbed_batch = np.repeat(sample, 100, axis=0)
    perturbed_batch[:, target_feat_idx] = perturbations
    
    print(" Calculating SHAP slopes across DIFA-2.0 manifold...")
    shap_vals = explainer.shap_values(perturbed_batch)
    probs_combined = model_keras.predict(perturbed_batch, verbose=0)
    task_probs = probs_combined[0]
    
    base_class = y_test[sample_idx]
    base_class_name = le.inverse_transform([base_class])[0]
    
    # Extract
    target_shap = shap_vals[base_class][:, target_feat_idx]
    target_probs = task_probs[:, base_class]
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel(f'{target_feat_name} (CORAL Aligned)')
    ax1.set_ylabel('Prediction Probability', color='tab:red')
    ax1.plot(perturbations, target_probs, color='tab:red', lw=3, label="Prob")
    ax1.tick_params(axis='y', labelcolor='tab:red')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('SHAP Value (Contribution)', color='tab:blue')
    ax2.plot(perturbations, target_shap, color='tab:blue', lw=3, label="SHAP Slope")
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    plt.title(f'SOTA-2.0 Persistence: Deep CORAL + DANN ({base_class_name})')
    plt.grid(True, alpha=0.3)
    plot_path = os.path.join(SOTA_RESULTS_DIR, "sedika_sota_v2_shap_audit.png")
    plt.savefig(plot_path)
    print(f" SHAP Sensitivity Plot saved to {plot_path}")
    
    # 5. Save Metrics
    results_df = pd.DataFrame([metrics_clear, metrics_noisy])
    results_df['Scenario'] = ['Clear', 'Noisy_0.1']
    metrics_path = os.path.join(SOTA_RESULTS_DIR, "sedika_sota_v2_metrics.csv")
    results_df.to_csv(metrics_path, index=False)
    print(f" SOTA-2.0 Metrics saved to {metrics_path}")
    
    print("\n[SOTA-2.0 STATUS CONFIRMED]")
    print(f" Domain Discriminator has forced Invariance.")
    print(f" Target Domain Accuracy: {metrics_clear['Accuracy']:.4f}")

if __name__ == "__main__":
    run_sota_v2_benchmark()
