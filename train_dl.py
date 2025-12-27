import pandas as pd
import numpy as np
import os
import time
import pickle
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Input
from tensorflow.keras.callbacks import EarlyStopping
from ml_utils import evaluate_model, robustness_test

# Configuration
OUTPUT_DIR = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/results"
MODEL_DIR = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/models"
for d in [OUTPUT_DIR, MODEL_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# Set seeds
tf.random.set_seed(42)
np.random.seed(42)

class DLModelWrapper:
    """Wrapper to make Keras models compatible with our ml_utils evaluation functions"""
    def __init__(self, model, reshape_required=False):
        self.model = model
        self.reshape_required = reshape_required # For CNN/RNN

    def predict(self, X):
        if self.reshape_required:
            # Reshape X to (samples, features, 1)
            X = X.values.reshape((X.shape[0], X.shape[1], 1))
        
        probs = self.model.predict(X, verbose=0)
        return np.argmax(probs, axis=1)

def load_data():
    base_dir = "c:/Users/mizal/.gemini/antigravity/scratch/iot_project_2/processed_data"
    training_set = pd.read_pickle(os.path.join(base_dir, "train_data.pkl"))
    val_set = pd.read_pickle(os.path.join(base_dir, "val_data.pkl"))
    test_set = pd.read_pickle(os.path.join(base_dir, "test_data.pkl"))
    
    X_train = training_set.drop(columns=['target'])
    y_train = training_set['target']
    
    X_val = val_set.drop(columns=['target'])
    y_val = val_set['target']
    
    X_test = test_set.drop(columns=['target'])
    y_test = test_set['target']
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def build_dnn(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_cnn(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim, 1)),
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_lstm(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim, 1)),
        LSTM(64),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def build_gru(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim, 1)),
        GRU(64),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_dl_models():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # 3D Data for CNN/RNN
    X_train_3d = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_3d = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    models_config = [
        ("DNN", build_dnn(input_dim, num_classes), False),
        ("CNN", build_cnn(input_dim, num_classes), True),
        ("LSTM", build_lstm(input_dim, num_classes), True),
        ("GRU", build_gru(input_dim, num_classes), True)
    ]
    
    results = []
    robustness_data = {}
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    print("\nStarting Deep Learning Training Loop...")
    
    for name, model, is_3d in models_config:
        print(f"\nProcessing {name}...")
        
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}.keras")
        train_time = 0
        
        # Check if model exists
        if os.path.exists(model_path):
            print(f"  Model found at {model_path}. Loading...")
            model = tf.keras.models.load_model(model_path)
        else:
            # Prepare Inputs
            if is_3d:
                train_X = X_train_3d
                val_X = X_val_3d
            else:
                train_X = X_train
                val_X = X_val
            
            # Subsample for RNNs (LSTM/GRU) due to CPU constraints
            if name in ["LSTM", "GRU"]:
                print(f"  Subsampling {name} training data to 50,000 samples for speed...")
                # Create indices
                indices = np.random.choice(train_X.shape[0], 50000, replace=False)
                train_X = train_X[indices]
                # Filter y_train as well - need to be careful with numpy/pandas mix
                # train_X is numpy array (via values or reshape), y_train is pandas Series
                y_sub = y_train.iloc[indices].values
                
                # Validation handled by EarlyStopping on full/partial? 
                # Let's keep Validation full or also subsample? 
                # Validation is small (18k), so it's fine.
            else:
                y_sub = y_train
                
            # Train
            t0 = time.time()
            model.fit(train_X, y_sub, 
                      validation_data=(val_X, y_val),
                      epochs=10, 
                      batch_size=128,
                      callbacks=[early_stop],
                      verbose=1)
            train_time = time.time() - t0
            print(f"  Training Time: {train_time:.2f}s")
            
            # Save Model (H5)
            model.save(model_path)
            
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Wrap for Evaluation
        wrapper = DLModelWrapper(model, reshape_required=is_3d)
        
        # Evaluate on Test Set
        metrics = evaluate_model(wrapper, X_test, y_test, name, OUTPUT_DIR)
        metrics['Training_Time_s'] = train_time
        metrics['Model_Size_MB'] = model_size_mb
        results.append(metrics)
        
        # Robustness Test
        print("  Running Robustness Test...")
        levels, accs = robustness_test(wrapper, X_test, y_test, name)
        robustness_data[name] = accs

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "dl_performance_metrics.csv"), index=False)
    
    # Save Robustness Data
    robust_df = pd.DataFrame(robustness_data, index=[f"Noise_{l}" for l in [0.01, 0.05, 0.1, 0.2]])
    robust_df.to_csv(os.path.join(OUTPUT_DIR, "dl_robustness.csv"))
    
    print("\nDL Training Complete. Results saved.")
    print(results_df[['Model', 'Accuracy', 'Latency_ms', 'Model_Size_MB']])

if __name__ == "__main__":
    train_dl_models()
