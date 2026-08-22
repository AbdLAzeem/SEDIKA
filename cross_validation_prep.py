import pandas as pd
import numpy as np
import os
import joblib
import re
from artifacts import load_artifact

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXTERNAL_DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "processed_external")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "processed_data")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Baseline Feature List (Top 25 selected during preprocessing)
BASELINE_FEATURES = [
    'id.resp_p', 'fwd_pkts_payload.avg', 'fwd_pkts_payload.tot', 'no', 'fwd_pkts_payload.min', 
    'flow_duration', 'fwd_pkts_payload.max', 'fwd_subflow_bytes', 'flow_iat.min', 'active.min', 
    'service', 'bwd_pkts_payload.max', 'flow_pkts_payload.avg', 'active.tot', 'flow_pkts_payload.max', 
    'fwd_PSH_flag_count', 'fwd_header_size_tot', 'fwd_iat.min', 'flow_iat.avg', 'fwd_last_window_size', 
    'flow_iat.std', 'fwd_init_window_size', 'active.max', 'flow_pkts_payload.min', 'bwd_subflow_bytes'
]

# Load original scaler and label encoder (manifest-validated)
SCALER = load_artifact(os.path.join(PROCESSED_DATA_DIR, "scaler.joblib"))
LABEL_ENCODER = load_artifact(os.path.join(PROCESSED_DATA_DIR, "label_encoder.joblib"))

# Label Taxonomy Mapping
LABEL_MAPPING = {
    # CICIoT2023
    'BenignTraffic': 'Thing_Speak',
    'DDoS-SlowLoris': 'DDOS_Slowloris',
    'DoS-SYN_Flood': 'DOS_SYN_Hping',
    'Recon-PortScan': 'NMAP_TCP_scan',
    'DictionaryBruteForce': 'Metasploit_Brute_Force_SSH',
    # IoT-23 (Standard labels usually benign or malicious with details)
    'benign': 'Thing_Speak',
    'DDoS': 'DDOS_Slowloris'
}

def map_ciciot2023(df):
    """Maps CICIoT2023 schema to Baseline."""
    print("Aligning CICIoT2023...")
    mapped = pd.DataFrame(index=df.index)
    
    # Feature Engineering/Mapping
    mapped['flow_duration'] = df['flow_duration']
    mapped['fwd_pkts_payload.avg'] = df['AVG'] # Heuristic: AVG as proxy
    mapped['fwd_pkts_payload.tot'] = df['Tot size']
    mapped['fwd_pkts_payload.min'] = df['Min']
    mapped['fwd_pkts_payload.max'] = df['Max']
    mapped['flow_iat.avg'] = df['IAT']
    mapped['fwd_PSH_flag_count'] = df['psh_flag_number']
    
    # Fill remaining required features with defaults or heuristic zeros if not present
    for col in BASELINE_FEATURES:
        if col not in mapped.columns:
            mapped[col] = 0.0
            
    # Label Alignment
    mapped['target_raw'] = df['label'].map(LABEL_MAPPING).fillna('Other')
    return mapped[BASELINE_FEATURES], mapped['target_raw']

def map_iot23(df):
    """Maps IoT-23 schema to Baseline."""
    print("Aligning IoT-23...")
    mapped = pd.DataFrame(index=df.index)
    
    # Feature Engineering/Mapping
    mapped['flow_duration'] = pd.to_numeric(df['duration'], errors='coerce').fillna(0)
    # Reconstruct volumetric features
    orig_pkts = pd.to_numeric(df['orig_pkts'], errors='coerce').fillna(1)
    orig_bytes = pd.to_numeric(df['orig_bytes'], errors='coerce').fillna(0)
    mapped['fwd_pkts_payload.avg'] = orig_bytes / orig_pkts
    mapped['fwd_pkts_payload.tot'] = orig_bytes
    mapped['id.resp_p'] = pd.to_numeric(df['id.resp_p'], errors='coerce').fillna(0)
    
    # History decoding for PSH flag (using history if available)
    if 'history' in df.columns:
        mapped['fwd_PSH_flag_count'] = df['history'].astype(str).apply(lambda x: 1 if 'P' in x else 0)
    
    # Service encoding
    if 'service' in df.columns:
        service_map = {'-': 0, 'dns': 1, 'http': 2, 'ssh': 3, 'ssl': 4, 'dhcp': 5}
        mapped['service'] = df['service'].map(service_map).fillna(0)

    # Fill remaining required features with zero if not found
    for col in BASELINE_FEATURES:
        if col not in mapped.columns:
            mapped[col] = 0.0

    # Identify label column (handle the specific spaced naming in IoT-23)
    composite_col = 'tunnel_parents   label   detailed-label'
    if composite_col in df.columns:
        splits = df[composite_col].astype(str).str.split()
        
        def map_label(row_splits):
            if len(row_splits) < 2:
                return 'Other'
            
            main_label = row_splits[1]
            detailed_label = row_splits[2] if len(row_splits) > 2 else '-'
            
            if main_label.lower() == 'benign':
                return 'Thing_Speak'
            
            # Specific malicious mappings based on detailed-label
            detailed_lower = detailed_label.lower()
            if 'portscan' in detailed_lower:
                return 'NMAP_TCP_scan'
            elif 'ddos' in detailed_lower:
                return 'DDOS_Slowloris'
            elif 'c&c' in detailed_lower:
                return 'Other' # Will be filtered
                
            return 'Other'
            
        mapped['target_raw'] = splits.apply(map_label)
    else:
        # Fallback to general search
        label_col = [c for c in df.columns if 'label' in c.lower()][0]
        mapped['target_raw'] = df[label_col].astype(str).str.strip().map(LABEL_MAPPING).fillna('Other')
        
    return mapped[BASELINE_FEATURES], mapped['target_raw']

def process_and_save(X, labels, name):
    print(f"Finalizing {name}...")
    # Clean data (NaN/Inf)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Scale using Baseline parameters
    X_scaled = SCALER.transform(X)
    X_df = pd.DataFrame(X_scaled, columns=BASELINE_FEATURES, index=X.index)
    
    # Filter only labels that exist in our Baseline Target Taxonomy
    mask = labels != 'Other'
    X_final = X_df[mask]
    y_final = labels[mask]
    
    # Encode labels
    y_encoded = LABEL_ENCODER.transform(y_final)
    
    # Save as Pickle
    final_df = X_final.copy()
    final_df['target'] = y_encoded
    output_path = os.path.join(OUTPUT_DIR, f"{name.lower()}_aligned.pkl")
    final_df.to_pickle(output_path)
    print(f" Saved to {output_path} (Samples: {len(final_df)})")

def main():
    # 1. CICIoT2023
    cic_path = os.path.join(EXTERNAL_DATA_DIR, "CICIoT2023-test.csv")
    if os.path.exists(cic_path):
        # Sample for speed if file is huge (approx 347MB)
        df_cic = pd.read_csv(cic_path).sample(50000, random_state=42)
        X, labels = map_ciciot2023(df_cic)
        process_and_save(X, labels, "CICIoT2023")

    # 2. IoT-23 (Merge multiple files)
    iot23_files = ["IOT-23-dataset17.csv", "IOT-23-dataset19.csv"]
    iot23_list = []
    print(f"Merging {len(iot23_files)} IoT-23 files...")
    for f in iot23_files:
        path = os.path.join(EXTERNAL_DATA_DIR, f)
        df_tmp = pd.read_csv(path, low_memory=False)
        iot23_list.append(df_tmp)
        
    if iot23_list:
        df_iot23 = pd.concat(iot23_list, ignore_index=True)
        X, labels = map_iot23(df_iot23)
        process_and_save(X, labels, "IoT23")

if __name__ == "__main__":
    main()
