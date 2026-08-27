"""
LSTM-based multi-label symptom classifier.
Architecture: Embedding -> Bidirectional LSTM -> Dense(sigmoid).
Bidirectionality helps capture context from both directions of a short
symptom description (e.g. negation appearing after the symptom word).
"""
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from app.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_TYPE = "lstm"
WEIGHTS_FILENAME = "lstm_model.keras"


def build_lstm_model(vocab_size: int, num_labels: int) -> tf.keras.Model:
    cfg = get_config()
    model = models.Sequential([
        layers.Input(shape=(cfg.MAX_SEQUENCE_LENGTH,)),
        layers.Embedding(input_dim=vocab_size, output_dim=cfg.EMBEDDING_DIM, mask_zero=True),
        layers.Bidirectional(layers.LSTM(cfg.LSTM_UNITS)),
        layers.Dropout(0.3),
        layers.Dense(num_labels, activation="sigmoid"),
    ], name="lstm_multilabel_classifier")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.DL_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc", multi_label=True), "binary_accuracy"],
    )
    return model


def train_lstm_model(model: tf.keras.Model, X_train, y_train, X_val, y_val, checkpoint_path: str):
    cfg = get_config()
    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=cfg.EARLY_STOPPING_PATIENCE, restore_best_weights=True),
        callbacks.ModelCheckpoint(checkpoint_path, monitor="val_loss", save_best_only=True),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.DL_EPOCHS,
        batch_size=cfg.DL_BATCH_SIZE,
        callbacks=cb,
        verbose=2,
    )
    return history


def predict_proba(model: tf.keras.Model, X: np.ndarray) -> np.ndarray:
    return model.predict(X, verbose=0)
