import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, Layer
from tensorflow.keras.models import Model
import os
from dataclasses import dataclass, field, asdict
from paths import PROCESSED_DIR, EXTERNAL_DIR, MODEL_DIR

# Use the SMOTE-balanced source pool so DIFA's 30k subsample is class-balanced
# (un-resampled, the subsample would be dominated by the majority class).
SOURCE_DATA = os.path.join(PROCESSED_DIR, "train_data_smote.pkl")
TARGET_DATA = os.path.join(EXTERNAL_DIR, "sedika_ciciot2023_coral.pkl")
SOURCE_MODEL_PATH = os.path.join(MODEL_DIR, "dnn.keras")
DIFA_V2_MODEL_PATH = os.path.join(MODEL_DIR, "sedika_difa_v2.keras")

# Layer names — captured here so the discriminator/backbone variable partition
# below is robust to Keras auto-numbering (the previous code matched
# `'dense_11' not in v.name`, which broke the moment a layer was reordered).
DISC_HIDDEN_NAME = "domain_hidden"
DISC_OUTPUT_NAME = "domain_output"
TASK_OUTPUT_NAME = "task_output"
GRL_NAME = "gradient_reversal"


# 1. Gradient Reversal Layer (GRL)
@tf.custom_gradient
def reverse_gradient(x, hp_lambda=1.0):
    def grad(dy):
        return -dy * hp_lambda, None
    return tf.identity(x), grad

class GradientReversal(Layer):
    def __init__(self, hp_lambda=1.0, **kwargs):
        super().__init__(**kwargs)
        self.hp_lambda = tf.Variable(hp_lambda, trainable=False, dtype=tf.float32)

    def call(self, x):
        return reverse_gradient(x, self.hp_lambda)


@dataclass
class LossWeights:
    """Explicit composition weights for the DIFA-2.2 multi-objective loss.

    Previously these were inlined as magic numbers (`+ 0.5 * loss_ent`,
    `+ current_lambda * loss_domain`). Pulling them out makes ablations
    trivial and per-epoch logging meaningful.
    """
    alpha_cls: float = 1.0          # task classification weight
    lambda_dann: float = 0.0        # adversarial domain weight (scheduled)
    gamma_entropy: float = 0.5      # target-prediction entropy minimisation

    def total(self, loss_task, loss_domain, loss_entropy):
        return (self.alpha_cls * loss_task
                + self.lambda_dann * loss_domain
                + self.gamma_entropy * loss_entropy)


def lambda_schedule(epoch: int, warmup: int = 10, ramp_epochs: int = 40,
                    lambda_max: float = 1.0) -> float:
    """0 during warmup, then linear ramp to lambda_max."""
    if epoch < warmup:
        return 0.0
    return min(lambda_max, (epoch - warmup) / float(ramp_epochs))


def partition_vars(model: Model):
    """Split trainable variables into (backbone, discriminator) lists by
    walking layers — no fragile string matching on auto-named tensors.
    """
    disc_layer_names = {DISC_HIDDEN_NAME, DISC_OUTPUT_NAME}
    disc_vars = []
    for layer in model.layers:
        if layer.name in disc_layer_names:
            disc_vars.extend(layer.trainable_variables)
    disc_var_ids = {id(v) for v in disc_vars}
    backbone_vars = [v for v in model.trainable_variables if id(v) not in disc_var_ids]
    return backbone_vars, disc_vars


# 2. Optimized Training Loop (DIFA-2.2)
def train_difa_v2(weights: LossWeights = None, *, epochs: int = 50,
                  save_path: str = None, verbose: bool = True,
                  source_n: int = 30000, target_n: int = 30000,
                  data_seed: int = 42):
    """Train DIFA-2.2 with the given loss weights.

    Returns the trained model so callers (e.g. ablation harness) can evaluate
    without reloading from disk. If `save_path` is None, defaults to
    DIFA_V2_MODEL_PATH for backwards-compatible CLI behaviour.
    """
    if save_path is None:
        save_path = DIFA_V2_MODEL_PATH
    if weights is None:
        weights = LossWeights()
    print("SEDIKA Phase 2: DIFA-2.2 Optimized Adversarial Marathon")

    # Load and Subset Data
    source_df = pd.read_pickle(SOURCE_DATA)
    target_df = pd.read_pickle(TARGET_DATA)

    # Fast-Track Stratified Sampling
    source_df = source_df.sample(n=source_n, random_state=data_seed)
    target_df = target_df.sample(n=target_n, random_state=data_seed)

    X_s = source_df.drop(columns=['target']).values
    # Remove Index 3 ('no' sequence index — spurious domain fingerprint per Section 3.6.3)
    # from BOTH source and target so the backbone input dim is 24 consistently.
    X_s = np.delete(X_s, 3, axis=1)
    y_s = source_df['target'].values
    X_t = target_df.drop(columns=['target']).values
    if X_t.shape[1] == 25:
        X_t = np.delete(X_t, 3, axis=1)

    target_classes = 34

    # 3. Backbone Slicing (Remove Index 3 weight vector)
    print(f" Slicing Backbone from {SOURCE_MODEL_PATH}...")
    source_model = tf.keras.models.load_model(SOURCE_MODEL_PATH)

    # Resolve the first two Dense layers by traversal rather than by string name.
    # Earlier versions of this file hard-coded 'dense_9' / 'dense_10' which broke
    # silently every time Keras's auto-numbering reset (e.g. after retraining
    # with class_weight). Walking by type is invariant to that.
    dense_layers = [l for l in source_model.layers
                    if isinstance(l, tf.keras.layers.Dense)]
    if len(dense_layers) < 2:
        raise RuntimeError(
            f"Expected at least 2 Dense layers in source model; found "
            f"{len(dense_layers)} ({[l.name for l in dense_layers]})"
        )
    first_dense, second_dense = dense_layers[0], dense_layers[1]
    print(f"  Resolved backbone: {first_dense.name} -> {second_dense.name}")

    old_weights, old_biases = first_dense.get_weights()
    # Delete index 3 (the 'no' feature)
    new_weights = np.delete(old_weights, 3, axis=0)

    # Build new Backbone for 24 inputs
    inputs_24 = Input(shape=(24,))
    x = Dense(128, activation='relu', name='dense_9_sliced')(inputs_24)
    x = Dropout(0.2, name='dropout_2_sliced')(x)
    x = Dense(64, activation='relu', name='dense_10_sliced')(x)
    x = Dropout(0.2, name='dropout_3_sliced')(x)

    backbone_sliced = Model(inputs=inputs_24, outputs=x)

    # Port over the weights (sliced first layer, intact second)
    backbone_sliced.get_layer('dense_9_sliced').set_weights([new_weights, old_biases])
    backbone_sliced.get_layer('dense_10_sliced').set_weights(second_dense.get_weights())

    # 4. Build DANN-2.2 Architecture
    features = backbone_sliced(inputs_24)
    task_logits = Dense(target_classes, activation='softmax', name=TASK_OUTPUT_NAME)(features)

    # Slower Discriminator (32 neurons) — explicit names so partition_vars() works
    grl_layer = GradientReversal(hp_lambda=0.0, name=GRL_NAME)(features)
    domain_hidden = Dense(32, activation='relu', name=DISC_HIDDEN_NAME)(grl_layer)
    domain_logits = Dense(1, activation='sigmoid', name=DISC_OUTPUT_NAME)(domain_hidden)

    model = Model(inputs=inputs_24, outputs=[task_logits, domain_logits])

    # 5. Differential Optimizers
    backbone_opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    disc_opt = tf.keras.optimizers.Adam(learning_rate=0.0001)  # 10x slower

    # Capture the backbone/discriminator variable split ONCE at build time.
    backbone_vars, disc_vars = partition_vars(model)
    print(f"  Partitioned vars -> backbone: {len(backbone_vars)}  discriminator: {len(disc_vars)}")

    # 6. Training Loop
    batch_size = 64

    # Per-epoch history for Figure 4 (convergence) and ablation reproducibility.
    history_rows = []

    print(f" Loss weights: {asdict(weights)} (lambda_dann is overwritten per epoch by schedule)")
    print(f" Starting High-Precision SOTA Marathon ({epochs} epochs)...")
    for epoch in range(epochs):
        # Lambda Scheduler: 1-10: 0.0, 11-50: linear ramp to 1.0
        weights.lambda_dann = lambda_schedule(epoch)
        model.get_layer(GRL_NAME).hp_lambda.assign(weights.lambda_dann)

        # Shuffle
        idx_s = np.random.permutation(len(X_s))
        idx_t = np.random.permutation(len(X_t))
        num_batches = len(X_s) // batch_size

        epoch_task_loss = 0.0
        epoch_domain_loss = 0.0
        epoch_entropy_loss = 0.0

        for i in range(num_batches):
            x_s_b = X_s[idx_s[i*batch_size:(i+1)*batch_size]].astype('float32')
            y_s_b = y_s[idx_s[i*batch_size:(i+1)*batch_size]]
            x_t_b = X_t[idx_t[i*batch_size:(i+1)*batch_size]].astype('float32')

            x_combined = np.vstack([x_s_b, x_t_b])
            domain_labels = np.hstack([np.zeros(batch_size), np.ones(batch_size)]).astype('float32')

            with tf.GradientTape(persistent=True) as tape:
                task_preds, domain_preds = model(x_combined, training=True)
                task_preds_t = task_preds[batch_size:]

                # Use CICO labels from aligned pkl for task loss
                y_t_b = target_df.iloc[idx_t[i*batch_size:(i+1)*batch_size]]['target'].values

                loss_task = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y_t_b, task_preds_t))
                loss_domain = tf.reduce_mean(tf.keras.losses.binary_crossentropy(domain_labels, tf.squeeze(domain_preds)))

                # Entropy minimization on target predictions
                loss_ent = -tf.reduce_mean(tf.reduce_sum(task_preds_t * tf.math.log(task_preds_t + 1e-10), axis=1))

                total_loss = weights.total(loss_task, loss_domain, loss_ent)

            backbone_opt.apply_gradients(zip(tape.gradient(total_loss, backbone_vars), backbone_vars))
            disc_opt.apply_gradients(zip(tape.gradient(total_loss, disc_vars), disc_vars))
            del tape

            epoch_task_loss += float(loss_task)
            epoch_domain_loss += float(loss_domain)
            epoch_entropy_loss += float(loss_ent)

        history_rows.append({
            "epoch": epoch + 1,
            "alpha_cls": weights.alpha_cls,
            "lambda_dann": weights.lambda_dann,
            "gamma_entropy": weights.gamma_entropy,
            "task_loss": epoch_task_loss / num_batches,
            "domain_loss": epoch_domain_loss / num_batches,
            "entropy_loss": epoch_entropy_loss / num_batches,
        })

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(
                f" Epoch {epoch+1:3d} | "
                f"a_cls={weights.alpha_cls:.2f} lam_dann={weights.lambda_dann:.2f} g_ent={weights.gamma_entropy:.2f} | "
                f"task={epoch_task_loss/num_batches:.4f} "
                f"domain={epoch_domain_loss/num_batches:.4f} "
                f"entropy={epoch_entropy_loss/num_batches:.4f}"
            )

    model.save(save_path)
    # Persist convergence history for Figure 4 and ablation post-mortems.
    hist_path = os.path.join(os.path.dirname(save_path),
                             f"{os.path.basename(save_path).replace('.keras', '_history.csv')}")
    pd.DataFrame(history_rows).to_csv(hist_path, index=False)
    print(f" DIFA-2.2 model saved to {save_path}")
    print(f" Convergence history saved to {hist_path}")
    return model

if __name__ == "__main__":
    train_difa_v2()
