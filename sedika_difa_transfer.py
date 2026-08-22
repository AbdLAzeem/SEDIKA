import pandas as pd
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
from sklearn.model_selection import train_test_split
from paths import EXTERNAL_DIR, MODEL_DIR

SOURCE_MODEL_PATH = os.path.join(MODEL_DIR, "dnn.keras")
TARGET_DATA_PATH = os.path.join(EXTERNAL_DIR, "sedika_ciciot2023_adaptive.pkl")
TARGET_MODEL_PATH = os.path.join(MODEL_DIR, "sedika_dnn_transfer.keras")

class NoisyValCallback(tf.keras.callbacks.Callback):
    """Evaluate the model on noisy validation data at each epoch.

    Noise is regenerated every epoch so the metric measures average-case
    perturbation tolerance, not memorisation of a single noise realisation.
    """
    def __init__(self, X_val, y_val, noise_level=0.05, seed=42):
        super().__init__()
        if hasattr(X_val, 'values'):
            X_val = X_val.values
        self.X_val_clean = X_val
        self.y_val = y_val
        self.noise_level = noise_level
        self.rng = np.random.default_rng(seed)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        noisy = self.X_val_clean + self.rng.normal(0, self.noise_level, self.X_val_clean.shape)
        loss, acc = self.model.evaluate(noisy, self.y_val, verbose=0)
        logs['val_noisy_accuracy'] = acc

def implement_transfer_learning():
    print("SEDIKA Phase 2: Domain-Invariant Transfer Learning (DIFA)")
    
    # 1. Load Adaptive Target Data
    df = pd.read_pickle(TARGET_DATA_PATH)
    X = df.drop(columns=['target']).values
    y = df['target'].values
    num_classes = len(np.unique(y))
    
    # 2. Stratified 10% Subset for Fine-Tuning
    X_tune, X_rem, y_tune, y_rem = train_test_split(X, y, train_size=0.10, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tune, y_tune, test_size=0.2, stratify=y_tune, random_state=42)
    
    print(f" Fine-tuning on {len(X_train)} samples across {num_classes} classes.")
    
    # 3. Load Source Backbone
    print(f" Loading backbone from {SOURCE_MODEL_PATH}...")
    source_model = tf.keras.models.load_model(SOURCE_MODEL_PATH)
    
    # Freeze the first few layers (Backbone)
    # Based on summary: dense_9, dropout_2, dense_10, dropout_3
    # We remove the last layer (dense_11)
    # The layers are accessable via index. 
    # Let's rebuild the model up to the last dropout layer.
    
    backbone_output = source_model.layers[-2].output # Output of dropout_3
    
    # 4. Head Recalibration
    new_head = Dense(num_classes, activation="softmax", name="sedika_head")(backbone_output)
    
    model = Model(inputs=source_model.input, outputs=new_head)
    
    # Freeze layers up to the new head
    for layer in model.layers[:-1]:
        layer.trainable = False
        
    print(" Backbone layers frozen. Head recalibrated.")
    
    # 5. Domain-Specific Fine-Tuning
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    noisy_cb = NoisyValCallback(X_val, y_val, noise_level=0.05) # Target noise awareness
    
    print(" Starting Robustness-Aware Fine-Tuning...")
    model.fit(X_train, y_train, 
              validation_data=(X_val, y_val),
              epochs=30, 
              batch_size=32, 
              callbacks=[early_stop, noisy_cb],
              verbose=1)
    
    # 6. Persistence
    model.save(TARGET_MODEL_PATH)
    print(f" SEDIKA Adaptive Model saved to {TARGET_MODEL_PATH}")

if __name__ == "__main__":
    implement_transfer_learning()
