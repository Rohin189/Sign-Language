# src/train.py
"""
Phase 2: train a baseline LSTM classifier on data/processed/dataset.npz.

Run from the project root:
    python -m src.train
"""

import json
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # hide TensorFlow info logs

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src import config


def load_dataset():
    """Load the processed splits and check them against config."""
    if not config.DATASET_PATH.exists():
        raise FileNotFoundError(
            f"{config.DATASET_PATH} not found. Run `python -m src.preprocess` first."
        )

    data = np.load(config.DATASET_PATH)
    signs = [str(s) for s in data["signs"]]
    if signs != config.SIGNS:
        raise ValueError(
            "Sign list in dataset.npz does not match config.SIGNS. "
            "Re-run `python -m src.preprocess`."
        )

    expected = (config.SEQUENCE_LENGTH, config.FEATURE_DIM_TOTAL)
    splits = {}
    for name in ("train", "val", "test"):
        X = data[f"X_{name}"].astype(np.float32)
        y = data[f"y_{name}"].astype(np.int64)
        if X.shape[1:] != expected:
            raise ValueError(f"X_{name} has shape {X.shape}; expected (N, {expected[0]}, {expected[1]}).")
        if len(X) != len(y):
            raise ValueError(f"X_{name} and y_{name} have different lengths.")
        if y.min() < 0 or y.max() >= config.NUM_CLASSES:
            raise ValueError(f"y_{name} contains labels outside 0..{config.NUM_CLASSES - 1}.")
        splits[name] = (X, y)
        print(f"  {name:<5}: {len(X):>4} samples")
    return splits


def build_model():
    """Masking -> 2x LSTM -> Dense -> softmax."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(config.SEQUENCE_LENGTH, config.FEATURE_DIM_TOTAL)),
        # Frames with no hands are all zeros; masking makes the LSTMs skip them.
        tf.keras.layers.Masking(mask_value=0.0),
        tf.keras.layers.LSTM(config.LSTM_UNITS, return_sequences=True),
        tf.keras.layers.Dropout(config.DROPOUT_RATE),
        tf.keras.layers.LSTM(config.LSTM_UNITS),
        tf.keras.layers.Dropout(config.DROPOUT_RATE),
        tf.keras.layers.Dense(config.DENSE_UNITS, activation="relu"),
        tf.keras.layers.Dense(config.NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def print_confusion_matrix(cm, labels):
    width = max(len(s) for s in labels)
    header = " " * (width + 2) + " ".join(f"{i:>3}" for i in range(len(labels)))
    print(header)
    for i, row in enumerate(cm):
        print(f"{labels[i]:>{width}} {i:>1} " + " ".join(f"{v:>3}" for v in row))
    print("  (rows = true sign, columns = predicted sign index)")


def main():
    tf.keras.utils.set_random_seed(config.RANDOM_SEED)

    print("Loading dataset...")
    splits = load_dataset()
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.MODEL_PATH),
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=2,
    )

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    config.TRAINING_HISTORY_PATH.write_text(json.dumps(hist, indent=2))
    best_epoch = int(np.argmin(hist["val_loss"])) + 1

    # Evaluate the best checkpoint on the held-out test set (used once, here).
    best_model = tf.keras.models.load_model(config.MODEL_PATH)
    train_loss, train_acc = best_model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
    test_loss, test_acc = best_model.evaluate(X_test, y_test, verbose=0)

    print("\n" + "=" * 60)
    print(f"Best epoch: {best_epoch} of {len(hist['loss'])}")
    print(f"  train  acc {train_acc:.3f}   loss {train_loss:.3f}")
    print(f"  val    acc {val_acc:.3f}   loss {val_loss:.3f}")
    print(f"  test   acc {test_acc:.3f}   loss {test_loss:.3f}")
    print("=" * 60)

    y_pred = np.argmax(best_model.predict(X_test, verbose=0), axis=1)
    labels = list(range(config.NUM_CLASSES))
    print("\nTest classification report:")
    print(classification_report(y_test, y_pred, labels=labels,
                                target_names=config.SIGNS, digits=3, zero_division=0))
    print("Test confusion matrix:")
    print_confusion_matrix(confusion_matrix(y_test, y_pred, labels=labels), config.SIGNS)

    print(f"\nModel saved to:   {config.MODEL_PATH}")
    print(f"History saved to: {config.TRAINING_HISTORY_PATH}")


if __name__ == "__main__":
    main()
