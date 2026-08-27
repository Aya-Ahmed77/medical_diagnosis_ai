"""
Transformer-based multi-label classifier, fine-tuned from a pretrained
BERT/BioBERT checkpoint (HuggingFace Transformers).

The pretrained checkpoint is configurable via TRANSFORMER_MODEL_NAME
(defaults to BioBERT, a medically-appropriate pretrained model per the
project's preference; falls back to TRANSFORMER_FALLBACK_MODEL_NAME if
the primary checkpoint cannot be downloaded, e.g. in offline/restricted
environments -- this fallback is logged loudly, never silent).
"""
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from app.config import get_config
from app.utils.logger import get_logger

logger = get_logger(__name__)

MODEL_TYPE = "transformer"


class SymptomMultiLabelDataset(Dataset):
    """torch Dataset wrapping tokenized texts + multi-hot label vectors."""

    def __init__(self, encodings, labels: np.ndarray):
        self.encodings = encodings
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def load_tokenizer_and_model(num_labels: int):
    cfg = get_config()
    model_name = cfg.TRANSFORMER_MODEL_NAME
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="multi_label_classification"
        )
        logger.info("Loaded transformer checkpoint '%s'.", model_name)
    except Exception as exc:  # noqa: BLE001 -- deliberately broad: network/HF errors vary
        logger.warning(
            "Could not load primary transformer '%s' (%s). Falling back to '%s'.",
            model_name, exc, cfg.TRANSFORMER_FALLBACK_MODEL_NAME,
        )
        model_name = cfg.TRANSFORMER_FALLBACK_MODEL_NAME
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels, problem_type="multi_label_classification"
        )
    return tokenizer, model, model_name


def tokenize_texts(tokenizer, texts: List[str]):
    cfg = get_config()
    return tokenizer(
        texts, truncation=True, padding="max_length", max_length=cfg.TRANSFORMER_MAX_LENGTH
    )


def fine_tune_transformer(
    tokenizer, model, train_texts, y_train, val_texts, y_val, output_dir: str
):
    cfg = get_config()
    train_encodings = tokenize_texts(tokenizer, train_texts)
    val_encodings = tokenize_texts(tokenizer, val_texts)

    train_dataset = SymptomMultiLabelDataset(train_encodings, y_train)
    val_dataset = SymptomMultiLabelDataset(val_encodings, y_val)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.TRANSFORMER_EPOCHS,
        per_device_train_batch_size=cfg.TRANSFORMER_BATCH_SIZE,
        per_device_eval_batch_size=cfg.TRANSFORMER_BATCH_SIZE,
        learning_rate=cfg.TRANSFORMER_LEARNING_RATE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()
    return trainer


def predict_proba(tokenizer, model, texts: List[str]) -> np.ndarray:
    cfg = get_config()
    model.eval()
    encodings = tokenize_texts(tokenizer, texts)
    input_ids = torch.tensor(encodings["input_ids"])
    attention_mask = torch.tensor(encodings["attention_mask"])
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits).numpy()
    return probs


def save_transformer(tokenizer, model, local_dir: str) -> str:
    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    logger.info("Saved transformer model+tokenizer to %s", local_dir)
    return local_dir


def load_transformer(local_dir: str, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(local_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        local_dir, num_labels=num_labels, problem_type="multi_label_classification"
    )
    return tokenizer, model
