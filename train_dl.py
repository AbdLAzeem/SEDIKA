import pandas as pd
import numpy as np
import os
import time
import pickle
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from ml_utils import evaluate_model, robustness_test
import matplotlib.pyplot as plt
from paths import PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR, PLOT_DIR, ensure_dirs

ensure_dirs(OUTPUT_DIR, MODEL_DIR, PLOT_DIR)

# Set seeds (overridable for multi-seed runs)
_SEED = int(os.environ.get("SEDIKA_SEED", 42))
tf.random.set_seed(_SEED)
np.random.seed(_SEED)

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
    # DL branch uses the un-resampled training pool; class imbalance is handled via class_weight.
    training_set = pd.read_pickle(os.path.join(PROCESSED_DIR, "train_data.pkl"))
    val_set = pd.read_pickle(os.path.join(PROCESSED_DIR, "val_data.pkl"))
    test_set = pd.read_pickle(os.path.join(PROCESSED_DIR, "test_data.pkl"))
    
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

# NoisyValCallback was removed — its mid-training noise probe duplicated the
# post-train sweep done by ml_utils.robustness_test, and adversarial_eval.py
# now provides FGSM/PGD curves which are the proper boundary-deterioration
# measure. Keeping it here would cost an extra forward pass per epoch for a
# metric the user never read.

def plot_learning_curves(history, name, output_dir):
    plt.figure(figsize=(15, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='s')

    plt.title(f'{name} - Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Val Loss', marker='s')
    plt.title(f'{name} - Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"learning_curve_{name.lower()}.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"  Learning curves saved to {plot_path}")

def train_dl_models():
    X_train, y_train, X_val, y_val, X_test, y_test = load_data()
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    
    # Standardize data if not already (though processed_data should be scaled)
    # 3D Data for CNN/RNN
    X_train_3d = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_val_3d = X_val.values.reshape((X_val.shape[0], X_val.shape[1], 1))
    
    models_config = [
        ("DNN", build_dnn, False),
        ("CNN", build_cnn, True),
        ("LSTM", build_lstm, True),
        ("GRU", build_gru, True)
    ]
    
    results = []
    robustness_data = [] # List of dicts for consistency

    # With epochs=50, give early stopping room to actually trigger.
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)

    # Class weighting replaces the SMOTE-on-DL pattern, which empirically hurts
    # generalisation in high-dim tabular settings.
    classes = np.unique(y_train)
    cw_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, cw_array)}

    print("\nStarting Deep Learning Training Loop...")
    
    for name, build_fn, is_3d in models_config:
        print(f"\nProcessing {name}...")
        
        model_path = os.path.join(MODEL_DIR, f"{name.lower()}.keras")
        train_time = 0
        
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
            indices = np.random.choice(train_X.shape[0], 50000, replace=False)
            train_X_sub = train_X[indices]
            y_sub = y_train.iloc[indices].values
        else:
            train_X_sub = train_X
            y_sub = y_train
            
        # Build and Train
        model = build_fn(input_dim, num_classes)
        t0 = time.time()
        history = model.fit(train_X_sub, y_sub,
                          validation_data=(val_X, y_val),
                          epochs=50,
                          batch_size=128,
                          callbacks=[early_stop, reduce_lr],
                          class_weight=class_weight,
                          verbose=1)
        train_time = time.time() - t0
        print(f"  Training Time: {train_time:.2f}s")
        
        # Plot History
        plot_learning_curves(history, name, PLOT_DIR)
        
        # Save Model
        model.save(model_path)
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        
        # Wrap for Evaluation
        wrapper = DLModelWrapper(model, reshape_required=is_3d)
        
        # Evaluate Best Metrics
        train_acc = history.history['accuracy'][-1]
        val_acc = history.history['val_accuracy'][-1]
        
        metrics = evaluate_model(wrapper, X_test, y_test, name, OUTPUT_DIR)
        metrics['Train_Accuracy'] = train_acc
        metrics['Val_Accuracy'] = val_acc
        metrics['Training_Time_s'] = train_time
        metrics['Model_Size_MB'] = model_size_mb
        
        # Fit status
        diff = train_acc - val_acc
        if diff > 0.05:
            metrics['Fit_Status'] = "Overfitting"
        elif train_acc < 0.7:
            metrics['Fit_Status'] = "Underfitting"
        else:
            metrics['Fit_Status'] = "Balanced"
            
        results.append(metrics)
        
        # Robustness Test
        print("  Running Robustness Test...")
        levels, robust_metrics = robustness_test(wrapper, X_test, y_test, name)
        for lvl, m in zip(levels, robust_metrics):
            m['Noise_Level'] = lvl
            m['Base_Model'] = name
            robustness_data.append(m)

    # Save Results
    pd.DataFrame(results).to_csv(os.path.join(OUTPUT_DIR, "dl_performance_metrics.csv"), index=False)
    pd.DataFrame(robustness_data).to_csv(os.path.join(OUTPUT_DIR, "dl_robustness.csv"), index=False)
    
    print("\nDL Training Complete. Results saved.")

if __name__ == "__main__":
    train_dl_models()
