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
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from paths import PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, ensure_dirs

# Overridable seed for multi-seed runs
_SEED = int(os.environ.get("SEDIKA_SEED", 42))
np.random.seed(_SEED)

ensure_dirs(OUTPUT_DIR, MODEL_DIR)

def load_data():
    print("Loading data...")
    # ML branch uses the SMOTE-balanced training pool (see preprocess_data.py).
    train = pd.read_pickle(os.path.join(PROCESSED_DIR, "train_data_smote.pkl"))
    val = pd.read_pickle(os.path.join(PROCESSED_DIR, "val_data.pkl"))
    test = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))
    
    X_train = train.drop(columns=['target'])
    y_train = train['target']
    
    X_val = val.drop(columns=['target'])
    y_val = val['target']
    
    X_test = test.drop(columns=['target'])
    y_test = test['target']
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def train_models():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    # Define Models and Hyperparameter Grids
    model_configs = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=_SEED),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=_SEED),
            "params": {
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10]
            }
        },
        "XGBoost": {
            "model": xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=_SEED),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            }
        },
        "LightGBM": {
            "model": lgb.LGBMClassifier(random_state=_SEED),
            "params": {
                "n_estimators": [50, 100],
                "learning_rate": [0.01, 0.1],
                "num_leaves": [31, 50]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": [3, 5],
                "weights": ["uniform", "distance"]
            }
        },
        "SVM": {
            "model": SVC(random_state=_SEED),
            "params": {
                "C": [1, 10],
                "kernel": ["rbf"]
            }
        }
    }
    
    results = []
    robustness_data = []
    tuning_log = []
    
    print("\nStarting Training & Tuning Loop...")
    
    # Optimization: Subsample for speed during development
    if len(X_train) > 20000:
        print(f"  Subsampling to 20,000 for faster demonstration...")
        X_train = X_train.sample(n=20000, random_state=_SEED)
        y_train = y_train.loc[X_train.index]

    for name, config in model_configs.items():
        print(f"\nProcessing {name}...")
        model = config["model"]
        param_grid = config["params"]
        
        # 1. Evaluate Before Tuning (Base Model)
        print(f"  Training Base Model...")
        model.fit(X_train, y_train)
        base_params = model.get_params()
        
        # 2. Hyperparameter Tuning
        print(f"  Performing Hyperparameter Tuning (RandomizedSearch)...")
        search = RandomizedSearchCV(model, param_grid, n_iter=2, cv=2, n_jobs=-1, random_state=_SEED)
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        
        tuning_log.append({
            "Model": name,
            "Params_Before": {k: base_params[k] for k in param_grid.keys() if k in base_params},
            "Params_After": best_params
        })
        
        # 3. Final Evaluation
        print(f"  Final Evaluation of Best Model...")
        t0 = time.time()
        best_model.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Metrics on Train, Val, Test
        train_acc = accuracy_score(y_train, best_model.predict(X_train))
        val_acc = accuracy_score(y_val, best_model.predict(X_val))
        
        metrics = evaluate_model(best_model, X_test, y_test, name, OUTPUT_DIR)
        metrics['Train_Accuracy'] = train_acc
        metrics['Val_Accuracy'] = val_acc
        metrics['Training_Time_s'] = train_time
        
        # Fit Discussion heuristic
        diff = train_acc - val_acc
        if diff > 0.05:
            fit_status = "Overfitting"
        elif train_acc < 0.7:
            fit_status = "Underfitting"
        else:
            fit_status = "Balanced"
        metrics['Fit_Status'] = fit_status
        
        results.append(metrics)
        
        # 4. Robustness Test (Best Model)
        print("  Running Robustness Test...")
        levels, robust_metrics = robustness_test(best_model, X_test, y_test, name)
        for lvl, m in zip(levels, robust_metrics):
            m['Noise_Level'] = lvl
            m['Base_Model'] = name
            robustness_data.append(m)
            
        # Save Best Model
        joblib.dump(best_model, os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}.pkl"))

    # Save Results
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "ml_performance_metrics.csv"), index=False)
    pd.DataFrame(robustness_data).to_csv(os.path.join(OUTPUT_DIR, "ml_robustness.csv"), index=False)
    pd.DataFrame(tuning_log).to_pickle(os.path.join(OUTPUT_DIR, "ml_tuning_log.pkl")) # Pickle for nested dicts
    
    # Save a readable tuning summary
    with open(os.path.join(OUTPUT_DIR, "ml_tuning_summary.txt"), "w") as f:
        for entry in tuning_log:
            f.write(f"Model: {entry['Model']}\n")
            f.write(f"  Before: {entry['Params_Before']}\n")
            f.write(f"  After: {entry['Params_After']}\n\n")

    print("\nTraining & Tuning Complete. Results saved.")

if __name__ == "__main__":
    train_models()
