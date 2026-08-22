import pandas as pd
import numpy as np
import os
import joblib
from scipy.linalg import sqrtm, inv
from paths import PROCESSED_DIR, EXTERNAL_DIR

# CORAL aligns *target* covariance to *source* covariance, so the source
# covariance must reflect the balanced class distribution the downstream
# DIFA classifier expects to see — use the SMOTE-balanced pool, not the raw
# class-imbalanced one (which would let the majority class dominate cov_s).
SOURCE_DATA_PATH = os.path.join(PROCESSED_DIR, "train_data_smote.pkl")
TARGET_RAW_PATH = os.environ.get(
    "SEDIKA_CICIOT_CSV",
    "CICIoT2023-test.csv",
)
OUTPUT_DIR = EXTERNAL_DIR

BASELINE_FEATURES = [
    'id.resp_p', 'fwd_pkts_payload.avg', 'fwd_pkts_payload.tot', 'no', 'fwd_pkts_payload.min', 
    'flow_duration', 'fwd_pkts_payload.max', 'fwd_subflow_bytes', 'flow_iat.min', 'active.min', 
    'service', 'bwd_pkts_payload.max', 'flow_pkts_payload.avg', 'active.tot', 'flow_pkts_payload.max', 
    'fwd_PSH_flag_count', 'fwd_header_size_tot', 'fwd_iat.min', 'flow_iat.avg', 'fwd_last_window_size', 
    'flow_iat.std', 'fwd_init_window_size', 'active.max', 'flow_pkts_payload.min', 'bwd_subflow_bytes'
]

def map_ciciot2023(df):
    mapped = pd.DataFrame(index=df.index)
    mapped['flow_duration'] = df['flow_duration']
    mapped['fwd_pkts_payload.avg'] = df['AVG']
    mapped['fwd_pkts_payload.tot'] = df['Tot size']
    mapped['fwd_pkts_payload.min'] = df['Min']
    mapped['fwd_pkts_payload.max'] = df['Max']
    mapped['flow_iat.avg'] = df['IAT']
    mapped['fwd_PSH_flag_count'] = df['psh_flag_number']
    for col in BASELINE_FEATURES:
        if col not in mapped.columns: mapped[col] = 0.0
    return mapped[BASELINE_FEATURES].values

def implement_coral_alignment():
    print("SEDIKA Phase 1: 2nd-Order Statistical Alignment (CORAL)")
    
    # 1. Load Source Data (RT-IoT)
    source_df = pd.read_pickle(SOURCE_DATA_PATH)
    X_source = source_df[BASELINE_FEATURES].values
    mu_s = np.mean(X_source, axis=0)
    cov_s = np.cov(X_source, rowvar=False) + np.eye(X_source.shape[1]) * 0.1 # Regularize
    
    # 2. Load Target Data (CICO)
    target_raw = pd.read_csv(TARGET_RAW_PATH)
    X_target = map_ciciot2023(target_raw)
    mu_t = np.mean(X_target, axis=0)
    cov_t = np.cov(X_target, rowvar=False) + np.eye(X_target.shape[1]) * 0.1 # Regularize
    
    print(" Calculating Matrix Square Roots for Covariance Alignment...")
    # 3. Compute CORAL Transform Matrix
    # Whiten Target, then Color as Source
    cov_t_inv_sqrt = inv(sqrtm(cov_t))
    cov_s_sqrt = sqrtm(cov_s)
    
    # Combined Alignment Matrix: A = C_t^-1/2 * C_s^1/2
    A = np.dot(cov_t_inv_sqrt, cov_s_sqrt)
    
    # 4. Apply Alignment
    # X_aligned = (X_t - mu_t) * A + mu_s
    X_target_centered = X_target - mu_t
    X_aligned = np.dot(X_target_centered, A) + mu_s
    
    # 5. Save Metadata and Aligned Data
    le = joblib.load(os.path.join(OUTPUT_DIR, "sedika_target_encoder.joblib")) # Reuse existing enc
    y_target = le.transform(target_raw['label'])
    
    aligned_df = pd.DataFrame(X_aligned, columns=BASELINE_FEATURES)
    aligned_df['target'] = y_target
    
    output_path = os.path.join(OUTPUT_DIR, "sedika_ciciot2023_coral.pkl")
    aligned_df.to_pickle(output_path)
    joblib.dump(A, os.path.join(OUTPUT_DIR, "sedika_coral_matrix.joblib"))
    
    print(f" 2nd-Order Alignment Complete. Aligned Dataset saved to {output_path}")

if __name__ == "__main__":
    implement_coral_alignment()
