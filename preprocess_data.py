import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
import joblib
from artifacts import save_artifact

# Ensure reproducible results
np.random.seed(42)

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    # Basic deduplication
    if df.duplicated().sum() > 0:
        print(f"Removing {df.duplicated().sum()} duplicates...")
        df = df.drop_duplicates()
    return df

def clean_data(df):
    print("Cleaning data (Infinite/Missing)...")
    # Replace infinite with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop rows with NaN
    initial_shape = df.shape
    df = df.dropna()
    if df.shape != initial_shape:
        print(f"Dropped {initial_shape[0] - df.shape[0]} rows containing NaN or Inf.")
    else:
        print("No missing or infinite values found to drop.")
    return df

def remove_high_correlation(X_train, X_test, threshold=0.98):
    print(f"Removing features with correlation > {threshold} (based on Train set)...")
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
        X_train_reduced = X_train.drop(columns=to_drop)
        X_test_reduced = X_test.drop(columns=to_drop)
        return X_train_reduced, X_test_reduced
    
    print("No highly correlated features found to drop.")
    return X_train, X_test

def feature_selection_rf(X_train, y_train, X_test, max_features=30):
    print("Performing Random Forest Feature Selection (on Train set)...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Keep top N
    top_indices = indices[:max_features]
    selected_columns = X_train.columns[top_indices]
    
    print(f"Selected Top {max_features} Features: {list(selected_columns)}")
    X_train_selected = X_train[selected_columns]
    X_test_selected = X_test[selected_columns]
    
    return X_train_selected, X_test_selected, rf

def preprocess_pipeline(file_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Load & Clean (Global)
    df = load_data(file_path)
    df = clean_data(df)
    
    # Separate Target
    target_col = 'Attack_type'
    if target_col not in df.columns:
        print(f"Error: Target column {target_col} not found.")
        return

    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Encode Target (Global is safe for label encoding target classes usually, but to be strict we could do it differently. 
    # For classes, fit on all known classes is standard.)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save label mapping with manifest
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"Target Mapping: {mapping}")
    save_artifact(
        le,
        os.path.join(output_dir, 'label_encoder.joblib'),
        feature_names=[target_col],
        kind="label_encoder",
        extra={"classes": list(map(str, le.classes_))},
    )

    # Encode Categorical Features (X)
    # Ideally fit on Train, transform Test.
    # Identifying categorical columns
    cat_cols = X.select_dtypes(include=['object']).columns
    print(f"Categorical columns found: {list(cat_cols)}")
    
    # 2. Split Data (Train/Val/Test)
    # Target: 70% Train, 15% Val, 15% Test
    print("Splitting data into Train (70%), Val (15%), and Test (15%)...")
    
    # First split: Train+Val (85%) and Test (15%)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded)
    
    # Second split: Train (approx 70% of total -> ~82.35% of temp) and Val (15% of total -> ~17.65% of temp)
    # 0.15 / 0.85 approx 0.1765
    val_size = 0.15 / 0.85
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp)
    
    # 3. Encoding Categorical Features
    # Fit on Train, vectorise Val/Test transforms via a dict map (unseen -> -1).
    unknown_label = -1
    for col in cat_cols:
        print(f"Encoding {col}...")
        le_col = LabelEncoder()
        X_train[col] = le_col.fit_transform(X_train[col])
        mapping = {cls: idx for idx, cls in enumerate(le_col.classes_)}
        X_val[col] = X_val[col].map(mapping).fillna(unknown_label).astype(int)
        X_test[col] = X_test[col].map(mapping).fillna(unknown_label).astype(int)

    # 4. Feature Selection: Remove High Correlation
    # Helper signature update required or just pass train/val/test?
    # Let's perform detection on Train, drop on all.
    print(f"Removing features with correlation > 0.98 (based on Train set)...")
    corr_matrix = X_train.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    
    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
        X_train = X_train.drop(columns=to_drop)
        X_val = X_val.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
    else:
        print("No highly correlated features found to drop.")

    # 5. Feature Selection: Tree-based (RF)
    # Fit on Train
    print("Performing Random Forest Feature Selection (on Train set)...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Keep top 25
    top_indices = indices[:25]
    selected_columns = X_train.columns[top_indices]
    
    print(f"Selected Top 25 Features: {list(selected_columns)}")
    X_train = X_train[selected_columns]
    X_val = X_val[selected_columns]
    X_test = X_test[selected_columns]

    # 6. Scaling
    print("Scaling data (StandardScaler)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Keep as DF
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    save_artifact(
        scaler,
        os.path.join(output_dir, 'scaler.joblib'),
        feature_names=list(X_train.columns),
        kind="standard_scaler",
    )

    # 7. Persist BOTH variants of the training pool.
    # train_data.pkl       -> un-resampled (consumed by DL + anomaly branches; DL uses class_weight)
    # train_data_smote.pkl -> SMOTE-balanced (consumed by classical ML branch)
    print("Saving un-resampled train pool...")
    train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    train_df['target'] = y_train.values if hasattr(y_train, 'values') else y_train
    train_df.to_pickle(os.path.join(output_dir, 'train_data.pkl'))

    print(f"Train class distribution before SMOTE: {dict(pd.Series(y_train).value_counts())}")
    print("Applying SMOTE for ML branch...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)
    print(f"Train class distribution after SMOTE: {dict(pd.Series(y_train_res).value_counts())}")
    train_smote_df = pd.DataFrame(X_train_res, columns=X_train.columns)
    train_smote_df['target'] = y_train_res
    train_smote_df.to_pickle(os.path.join(output_dir, 'train_data_smote.pkl'))

    # Val
    val_df = pd.DataFrame(X_val_scaled, columns=X_val.columns)
    val_df['target'] = y_val
    val_df.to_pickle(os.path.join(output_dir, 'val_data.pkl'))

    # Test
    test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    test_df['target'] = y_test
    test_df.to_pickle(os.path.join(output_dir, 'test_data.pkl'))

    print(
        f"Saved train_data.pkl ({train_df.shape}), "
        f"train_data_smote.pkl ({train_smote_df.shape}), "
        f"val_data.pkl ({val_df.shape}), test_data.pkl ({test_df.shape})"
    )

if __name__ == "__main__":
    from paths import RAW_CSV, PROCESSED_DIR
    preprocess_pipeline(RAW_CSV, PROCESSED_DIR)
