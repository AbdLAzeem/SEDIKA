import pandas as pd
import numpy as np
import os
import joblib
from artifacts import load_artifact

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")
EXTERNAL_DATA_DIR = os.path.join(BASE_DIR, "processed_external")

BASELINE_FEATURES = [
    'id.resp_p', 'fwd_pkts_payload.avg', 'fwd_pkts_payload.tot', 'no', 'fwd_pkts_payload.min', 
    'flow_duration', 'fwd_pkts_payload.max', 'fwd_subflow_bytes', 'flow_iat.min', 'active.min', 
    'service', 'bwd_pkts_payload.max', 'flow_pkts_payload.avg', 'active.tot', 'flow_pkts_payload.max', 
    'fwd_PSH_flag_count', 'fwd_header_size_tot', 'fwd_iat.min', 'flow_iat.avg', 'fwd_last_window_size', 
    'flow_iat.std', 'fwd_init_window_size', 'active.max', 'flow_pkts_payload.min', 'bwd_subflow_bytes'
]

def analyze_shift():
    # Load Baseline stats (Heuristic: we use a sample or the scaler's mean/scale)
    scaler = load_artifact(os.path.join(PROCESSED_DATA_DIR, "scaler.joblib"))
    baseline_means = pd.Series(scaler.mean_, index=BASELINE_FEATURES)
    
    results = {"baseline_mean": baseline_means}
    
    # Load External
    for ext_file in os.listdir(EXTERNAL_DATA_DIR):
        if ext_file.endswith(".pkl"):
            name = ext_file.split("_")[0].upper()
            df = pd.read_pickle(os.path.join(EXTERNAL_DATA_DIR, ext_file))
            X = df.drop(columns=['target'])
            # Note: The data in .pkl is ALREADY SCALED by the baseline scaler in cross_validation_prep.py
            # So if means are far from 0, it indicates shift.
            results[f"{name}_mean_scaled"] = X.mean()
            results[f"{name}_std_scaled"] = X.std()

    comparison = pd.DataFrame(results)
    comparison.to_csv(os.path.join(BASE_DIR, "results", "domain_shift_analysis.csv"))
    print("Domain shift analysis saved to results/domain_shift_analysis.csv")
    
    # Check for constant features or extreme outliers
    print("\nFeature Shift Insights:")
    for name in [c for c in comparison.columns if "mean_scaled" in c]:
        shift_count = (comparison[name].abs() > 1.0).sum()
        print(f" {name}: {shift_count}/{len(BASELINE_FEATURES)} features have mean shift > 1 std dev")

if __name__ == "__main__":
    analyze_shift()
