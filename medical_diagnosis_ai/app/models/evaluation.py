"""
Evaluation metrics shared by all four models so comparisons are
apples-to-apples.

EMERGENCY CASES DEFINITION (documented per requirement 11):
A test example is treated as an "emergency case" if the TRUE condition's
`warnings` text (as stored in MongoDB) contains one of cfg.EMERGENCY_KEYWORDS
(case-insensitive substring match), e.g. "999", "A&E", "immediately",
"life-threatening". This is a text-heuristic proxy over real NHS warning
content -- not a clinical severity judgement -- and is documented as a
limitation in the experimentation report.

"Emergency recall" = of all TRUE-positive labels that belong to an
emergency condition, what fraction did the model surface within its
top PREDICTION_TOP_N ranked predictions for that example. This measures
whether the system fails to flag genuinely urgent conditions, which
matters more than raw accuracy for safety.
"""
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

from app.config import get_config


@dataclass
class EvaluationMetrics:
    precision_micro: float
    recall_micro: float
    f1_micro: float
    top_1_accuracy: float
    top_3_accuracy: float
    top_5_accuracy: float
    emergency_recall: float
    num_emergency_examples: int

    def to_dict(self) -> Dict:
        return asdict(self)


def is_emergency_condition(warnings_text: str) -> bool:
    cfg = get_config()
    if not warnings_text:
        return False
    lowered = warnings_text.lower()
    return any(kw.lower() in lowered for kw in cfg.EMERGENCY_KEYWORDS)


def top_k_accuracy(y_true: np.ndarray, y_proba: np.ndarray, k: int) -> float:
    """Fraction of examples where at least one true label is within the
    model's top-k highest-probability predictions."""
    n = y_true.shape[0]
    if n == 0:
        return 0.0
    hits = 0
    top_k_idx = np.argsort(-y_proba, axis=1)[:, :k]
    for i in range(n):
        true_idx = set(np.where(y_true[i] == 1)[0])
        if true_idx & set(top_k_idx[i]):
            hits += 1
    return hits / n


def emergency_recall(
    y_true: np.ndarray, y_proba: np.ndarray, label_names: List[str],
    emergency_flags: List[bool], top_n: int = None,
) -> Dict:
    """emergency_flags[j] is True if label_names[j] is an emergency
    condition (see is_emergency_condition). Recall is computed only over
    the emergency-condition positive labels in the test set."""
    cfg = get_config()
    top_n = top_n or cfg.PREDICTION_TOP_N
    emergency_idx = {j for j, flag in enumerate(emergency_flags) if flag}
    if not emergency_idx:
        return {"emergency_recall": 0.0, "num_emergency_examples": 0}

    top_n_idx = np.argsort(-y_proba, axis=1)[:, :top_n]
    total_emergency_positives = 0
    captured = 0
    for i in range(y_true.shape[0]):
        true_idx = set(np.where(y_true[i] == 1)[0])
        true_emergency = true_idx & emergency_idx
        if not true_emergency:
            continue
        total_emergency_positives += len(true_emergency)
        captured += len(true_emergency & set(top_n_idx[i]))

    recall = captured / total_emergency_positives if total_emergency_positives else 0.0
    return {"emergency_recall": recall, "num_emergency_examples": total_emergency_positives}


def evaluate_predictions(
    y_true: np.ndarray, y_proba: np.ndarray, label_names: List[str],
    emergency_flags: List[bool], threshold: float = 0.5,
) -> EvaluationMetrics:
    y_pred = (y_proba >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    top1 = top_k_accuracy(y_true, y_proba, 1)
    top3 = top_k_accuracy(y_true, y_proba, 3)
    top5 = top_k_accuracy(y_true, y_proba, 5)

    er = emergency_recall(y_true, y_proba, label_names, emergency_flags)

    return EvaluationMetrics(
        precision_micro=float(precision),
        recall_micro=float(recall),
        f1_micro=float(f1),
        top_1_accuracy=float(top1),
        top_3_accuracy=float(top3),
        top_5_accuracy=float(top5),
        emergency_recall=float(er["emergency_recall"]),
        num_emergency_examples=int(er["num_emergency_examples"]),
    )
