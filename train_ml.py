import pandas as pd
import numpy as np
import os
import time
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from ml_utils import evaluate_model, robustness_test

# Configuration
OUTPUT_DIR = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/results"
MODEL_DIR = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/models"
for d in [OUTPUT_DIR, MODEL_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def load_data():
    base_dir = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/processed_data"
    print("Loading data...")
    train = pd.read_pickle(os.path.join(base_dir, "train_data.pkl"))
    val = pd.read_pickle(os.path.join(base_dir, "val_data.pkl"))
    test = pd.read_pickle(os.path.join(base_dir, "test_data.pkl"))
    
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    
    X_val = val.drop(columns=['target'])
    y_val = val['target']
    
    X_test = test.drop(columns=['target'])
    y_test = test['target']
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_models():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Define Models
    # Using specific params to speed up training for demo purposes where appropriate, 
    # but keeping them robust enough for good results.
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1, random_state=42),
        "LightGBM": lgb.LGBMClassifier(n_jobs=-1, random_state=42),
        # SVM is computationally expensive on large datasets (800k rows). 
        # We will use a subset for training or LinearSVC for speed? 
        # Let's try RBF with a smaller max_iter or subset.
        # Actually for 800k rows, SVM RBF will take forever. 
        # STRATEGY: Train SVM on a sample (e.g. 50k) or skip.
        # Let's train on 20% of training data for SVM to be realistic timely.
        "SVM": SVC(kernel='rbf', max_iter=1000, random_state=42) 
    }
    
    results = []
    robustness_data = {}
    
    print("\nStarting Training Loop...")
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train
        t0 = time.time()
        if name == "SVM":
            # Subsample for SVM
            print("  (Subsampling for SVM speed...)")
            X_sub = X_train.sample(n=50000, random_state=42)
            y_sub = y_train.loc[X_sub.index]
            model.fit(X_sub, y_sub)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - t0
        print(f"  Training Time: {train_time:.2f}s")
        
        # Save Model
        model_path = os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}.pkl")
        joblib.dump(model, model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Evaluate on Test Set
        metrics = evaluate_model(model, X_test, y_test, name, OUTPUT_DIR)
        metrics['Training_Time_s'] = train_time
        metrics['Model_Size_MB'] = model_size_mb
        results.append(metrics)
        
        # Robustness Test
        print("  Running Robustness Test...")
        levels, accs = robustness_test(model, X_test, y_test, name)
        robustness_data[name] = accs

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "ml_performance_metrics.csv"), index=False)
    
    # Save Robustness Data
    robust_df = pd.DataFrame(robustness_data, index=[f"Noise_{l}" for l in [0.01, 0.05, 0.1, 0.2]])
    robust_df.to_csv(os.path.join(OUTPUT_DIR, "ml_robustness.csv"))
    
    print("\nTraining Complete. Results saved.")
    print(results_df[['Model', 'Accuracy', 'Latency_ms', 'Model_Size_MB']])

if __name__ == "__main__":
    train_models()
