"""
Dataset construction for model training.

PROBLEM FORMULATION (documented per project requirement 7)
------------------------------------------------------------
The PDF explicitly requires multi-label prediction for the deep learning
models (a symptom description can plausibly match more than one
condition). We therefore formulate this as MULTI-LABEL classification:

    input  = free-text symptom description
    output = a binary vector over all known conditions (1 = plausible
             match), not a single mutually-exclusive class.

WHY THE RAW SCRAPED DATA NEEDS AUGMENTATION
------------------------------------------------------------
Each scraped NHS Inform condition page gives us exactly ONE (condition ->
its own symptom list) pair. That is naturally a ONE-example-per-class
dataset, which is useless for training a generalizing classifier and
provides no multi-label overlap at all.

To get a trainable multi-label dataset we generate additional training
EXAMPLES per condition by resampling combinations of that condition's own
scraped symptom phrases (e.g. subsets of {fever, sore throat, headache}
instead of always all three together). This:

  * uses ONLY real scraped symptom phrases as the vocabulary/content --
    no invented medical facts, no invented symptoms,
  * creates plausible partial-symptom presentations (people rarely report
    every listed symptom for a condition),
  * is clearly and permanently marked with is_synthetic=True on every
    generated row so scraped vs. synthetic rows are never conflated.

Multi-label overlap is introduced by: for a configurable fraction of
generated examples, symptom phrases from a second, textually similar
condition are mixed in (based on shared symptom-phrase overlap), and
BOTH conditions are marked as positive labels for that example. This
models the realistic case of overlapping/co-occurring symptom profiles
(e.g. "sore throat" appears under multiple conditions) without fabricating
any medical claim -- every phrase used is one NHS Inform actually
associated with that condition.

This strategy and its limitations are also written up in
reports/experimentation_report.md.
"""
import itertools
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from app.config import get_config
from app.preprocessing.text_cleaning import build_model_input_text
from app.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetRow:
    text: str
    labels: List[str]           # multi-label: one or more condition names
    is_synthetic: bool
    source_conditions: List[str] = field(default_factory=list)  # provenance


def _symptom_powerset_samples(symptoms: List[str], n_samples: int, rng: random.Random) -> List[List[str]]:
    """Return up to n_samples non-empty subsets of `symptoms`, biased
    toward partial (not full) subsets, without materializing the whole
    powerset for long lists."""
    cfg = get_config()
    symptoms = [s for s in symptoms if s]
    if not symptoms:
        return []
    if len(symptoms) <= 4:
        all_subsets = [
            list(c) for r in range(1, len(symptoms) + 1)
            for c in itertools.combinations(symptoms, r)
        ]
        rng.shuffle(all_subsets)
        return all_subsets[:n_samples]

    subsets = []
    for _ in range(n_samples):
        k = rng.randint(max(1, cfg.MIN_SYMPTOM_TOKENS - 1), len(symptoms))
        subsets.append(rng.sample(symptoms, k))
    return subsets


def _condition_similarity(a_symptoms: List[str], b_symptoms: List[str]) -> float:
    """Jaccard overlap between two conditions' symptom phrase sets --
    used only to decide which conditions are plausible co-labels."""
    set_a, set_b = set(a_symptoms), set(b_symptoms)
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def build_multilabel_dataset(
    conditions: List[Dict],
    synthetic_per_condition: int = None,
    multilabel_mix_fraction: float = 0.25,
    seed: int = None,
) -> Tuple[List[DatasetRow], List[str]]:
    """Build a multi-label training dataset from scraped condition documents.

    Args:
        conditions: list of condition dicts as stored in MongoDB
            (each with 'condition' and 'symptoms').
        synthetic_per_condition: number of generated rows per condition.
        multilabel_mix_fraction: fraction of generated rows that get a
            second, overlapping condition mixed in as an additional label.
        seed: RNG seed for reproducibility.

    Returns:
        (rows, label_space) where label_space is the sorted list of all
        condition names (the fixed output vocabulary for the classifiers).
    """
    cfg = get_config()
    synthetic_per_condition = synthetic_per_condition or cfg.SYNTHETIC_EXAMPLES_PER_CONDITION
    rng = random.Random(seed if seed is not None else cfg.RANDOM_SEED)

    label_space = sorted({c["condition"] for c in conditions if c.get("condition")})
    rows: List[DatasetRow] = []

    # One row per condition using its FULL scraped symptom list -- this is
    # the one genuinely "real" (non-synthetic) example per class.
    for cond in conditions:
        symptoms = cond.get("symptoms") or []
        if not symptoms:
            continue
        text = build_model_input_text(symptoms)
        if not text:
            continue
        rows.append(DatasetRow(
            text=text, labels=[cond["condition"]], is_synthetic=False,
            source_conditions=[cond["condition"]],
        ))

    # Synthetic partial-symptom rows, with occasional multi-label mixing.
    for i, cond in enumerate(conditions):
        symptoms = cond.get("symptoms") or []
        if len(symptoms) < 1:
            continue
        subsets = _symptom_powerset_samples(symptoms, synthetic_per_condition, rng)

        for subset in subsets:
            labels = [cond["condition"]]
            source = [cond["condition"]]
            combined_symptoms = list(subset)

            if rng.random() < multilabel_mix_fraction and len(conditions) > 1:
                # find another condition with meaningful symptom overlap
                candidates = [
                    other for j, other in enumerate(conditions)
                    if j != i and other.get("symptoms")
                    and _condition_similarity(symptoms, other["symptoms"]) > 0.0
                ]
                if candidates:
                    other = rng.choice(candidates)
                    other_subset = _symptom_powerset_samples(other["symptoms"], 1, rng)
                    if other_subset:
                        combined_symptoms += other_subset[0]
                        labels.append(other["condition"])
                        source.append(other["condition"])

            text = build_model_input_text(combined_symptoms)
            if not text:
                continue
            rows.append(DatasetRow(
                text=text, labels=sorted(set(labels)), is_synthetic=True,
                source_conditions=source,
            ))

    rng.shuffle(rows)
    logger.info(
        "Built dataset: %d rows (%d real, %d synthetic) across %d condition labels.",
        len(rows), sum(1 for r in rows if not r.is_synthetic),
        sum(1 for r in rows if r.is_synthetic), len(label_space),
    )
    return rows, label_space


def train_val_test_split(rows: List[DatasetRow], seed: int = None):
    """Shuffle and split rows according to cfg.TRAIN_SPLIT/VAL_SPLIT/TEST_SPLIT."""
    cfg = get_config()
    rng = random.Random(seed if seed is not None else cfg.RANDOM_SEED)
    shuffled = rows[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * cfg.TRAIN_SPLIT)
    n_val = int(n * cfg.VAL_SPLIT)

    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test
