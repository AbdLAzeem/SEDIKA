"""
Run this ONCE locally before pushing to Streamlit Community Cloud.
Extracts a 300-row stratified sample from processed_data/ into demo_data/
so the dashboard runs without the full 50 MB processed dataset.

    python build_demo_data.py
"""
import os, shutil
import pandas as pd
import joblib

SRC = "processed_data"
DST = "demo_data"

assert os.path.isdir(SRC), f"Run preprocess_data.py first — '{SRC}' not found."
os.makedirs(DST, exist_ok=True)

# Stratified 300-row sample of test_data
test_df = pd.read_pickle(os.path.join(SRC, "test_data.pkl"))
sample  = (test_df.groupby("target", group_keys=False)
                  .apply(lambda g: g.sample(min(len(g), 15), random_state=42)))
sample  = sample.sample(frac=1, random_state=42).reset_index(drop=True)
sample.to_pickle(os.path.join(DST, "test_data.pkl"))
print(f"test_data.pkl -> {len(sample)} rows  ({len(sample.columns)} cols)")

# Copy artefacts verbatim
for fname in ("scaler.joblib", "label_encoder.joblib"):
    shutil.copy2(os.path.join(SRC, fname), os.path.join(DST, fname))
    print(f"Copied  {fname}")

print(f"\ndemo_data/ ready ({os.path.getsize(os.path.join(DST,'test_data.pkl'))//1024} KB test set)")
