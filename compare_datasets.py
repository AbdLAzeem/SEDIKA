import pandas as pd
import numpy as np
import os

def load_processed(filepath):
    return pd.read_pickle(filepath)

def load_original(filepath):
    return pd.read_csv(filepath)

def compare():
    base_dir = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2"
    orig_path = os.path.join(base_dir, "RT_IOT2022.csv")
    train_path = os.path.join(base_dir, "processed_data", "train_data.pkl")
    test_path = os.path.join(base_dir, "processed_data", "test_data.pkl")
    
    print("Loading datasets...")
    df_orig = load_original(orig_path)
    df_train = load_processed(train_path)
    df_test = load_processed(test_path)
    
    print("\n--- Shape Comparison ---")
    print(f"Original: {df_orig.shape}")
    print(f"Processed Train: {df_train.shape} (Includes SMOTE)")
    print(f"Processed Test:  {df_test.shape}")
    
    print("\n--- Feature Comparison ---")
    orig_cols = set(df_orig.columns)
    train_cols = set(df_train.columns)
    print(f"Original Features ({len(orig_cols)}): {list(df_orig.columns)[:5]} ...")
    print(f"Final Features ({len(train_cols)}): {list(df_train.columns)}")
    print(f"Features Removed: {len(orig_cols - train_cols)}")
    
    print("\n--- Class Distribution (Top 5) ---")
    print("Original Target ('Attack_type'):")
    print(df_orig['Attack_type'].value_counts().head())
    
    print("\nProcessed Train Target ('target'):")
    print(df_train['target'].value_counts().head())
    
    print("\nProcessed Test Target ('target'):")
    print(df_test['target'].value_counts().head())
    
    print("\n--- Data Sample (First row) ---")
    print("\nOriginal (First 3 cols):")
    print(df_orig.iloc[0, :3])
    print("\nProcessed Train (First 3 cols):")
    print(df_train.iloc[0, :3])

if __name__ == "__main__":
    compare()
