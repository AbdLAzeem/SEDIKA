import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import joblib
from paths import EXTERNAL_DIR, MODEL_DIR

SOURCE_AE_PATH = os.path.join(MODEL_DIR, "ae_model.keras")
TARGET_DATA_PATH = os.path.join(EXTERNAL_DIR, "sedika_ciciot2023_adaptive.pkl")
TARGET_AE_PATH = os.path.join(MODEL_DIR, "sedika_ae_adapted.keras")
META_PATH = os.path.join(MODEL_DIR, "sedika_ae_threshold.joblib")

# Anomaly tier operating point. 0.005 = 0.5% benign-flow false positive rate.
# A static 95th-percentile threshold guarantees a 5% FPR by construction, which
# floods the alert queue at realistic IoT volumes — calibrate to a budget instead.
DEFAULT_FPR_BUDGET = 0.005


def calibrate_threshold(ae, X_calib, fpr_budget):
    """Pick tau so the empirical FPR on a held-out benign set <= fpr_budget."""
    recon = ae.predict(X_calib, verbose=0)
    mse = np.mean(np.power(X_calib - recon, 2), axis=1)
    # The (1 - budget) quantile is the smallest tau that keeps benign FPR within budget.
    tau = float(np.quantile(mse, 1.0 - fpr_budget))
    achieved_fpr = float(np.mean(mse > tau))
    return tau, achieved_fpr, mse


def implement_ae_adaptation(fpr_budget: float = DEFAULT_FPR_BUDGET):
    print("SEDIKA Phase 3: Unsupervised Manifold Alignment (The Fallback)")

    # 1. Load Aligned Target Data
    df = pd.read_pickle(TARGET_DATA_PATH)

    # 2. Benign Recalibration (Identify and Extract Normal Traffic)
    # Based on sedika_bridge, label 1 corresponds to BenignTraffic
    benign_df = df[df['target'] == 1]
    X_benign = benign_df.drop(columns=['target']).values
    print(f" Extracted {len(X_benign)} benign samples from target domain for recalibration.")

    # 2a. Split benign into fit/calibration — calibrating on the fit data biases tau low.
    X_fit, X_calib = train_test_split(X_benign, test_size=0.2, random_state=42)
    print(f"  Fit pool: {len(X_fit)}  |  Calibration pool: {len(X_calib)}")

    # 3. Backbone Warm-Start
    print(f" Loading source Autoencoder from {SOURCE_AE_PATH}...")
    ae = load_model(SOURCE_AE_PATH)

    # 4. Manifold Refitting (Epochs=5 for warm start stability)
    print(" Refitting Manifold on new environment noise floor...")
    ae.fit(X_fit, X_fit,
           epochs=5,
           batch_size=64,
           shuffle=True,
           verbose=1)

    # 5. FPR-budgeted threshold calibration on held-out benign data
    tau, achieved_fpr, _ = calibrate_threshold(ae, X_calib, fpr_budget)
    print(f" Threshold (tau) calibrated to FPR budget {fpr_budget:.4f}")
    print(f"   tau = {tau:.6f}  |  achieved benign FPR on calib set = {achieved_fpr:.4f}")

    # 6. Persistence — keep the budget alongside tau so re-fits stay comparable.
    ae.save(TARGET_AE_PATH)
    joblib.dump({
        "threshold": tau,
        "fpr_budget": fpr_budget,
        "achieved_fpr_calib": achieved_fpr,
        "calib_n": len(X_calib),
    }, META_PATH)
    print(f" Adapted Autoencoder saved to {TARGET_AE_PATH}")

if __name__ == "__main__":
    implement_ae_adaptation()
