from app.preprocessing.text_cleaning import (
    clean_text, normalize_symptom_phrase, build_model_input_text,
    split_free_text_into_clauses,
)
from app.preprocessing.dataset_builder import build_multilabel_dataset, train_val_test_split


def test_clean_text_lowercases_and_strips_punctuation():
    assert clean_text("Fever, Chills!!") == "fever chills"


def test_normalize_symptom_phrase_removes_safe_stopwords_only():
    result = normalize_symptom_phrase("Pain in the chest")
    assert "pain" in result and "chest" in result
    assert " the " not in f" {result} "


def test_build_model_input_text_dedupes_and_joins():
    text = build_model_input_text(["Fever", "fever", "Sore throat"])
    assert text.count("fever") == 1
    assert "sore throat" in text


def test_split_free_text_into_clauses():
    clauses = split_free_text_into_clauses("I have a fever, sore throat and headache.")
    assert "fever" in clauses[1] or any("fever" in c for c in clauses)
    assert len(clauses) >= 2


SAMPLE_CONDITIONS = [
    {
        "condition": "Asthma",
        "symptoms": ["wheezing", "shortness of breath", "chest tightness"],
        "warnings": "Call 999 if severe.",
    },
    {
        "condition": "Common cold",
        "symptoms": ["sore throat", "runny nose", "sneezing"],
        "warnings": "See a pharmacist if symptoms worsen.",
    },
]


def test_build_multilabel_dataset_produces_real_and_synthetic_rows():
    rows, label_space = build_multilabel_dataset(SAMPLE_CONDITIONS, synthetic_per_condition=5, seed=1)
    assert set(label_space) == {"Asthma", "Common cold"}
    assert any(not r.is_synthetic for r in rows)
    assert any(r.is_synthetic for r in rows)
    for r in rows:
        assert all(label in label_space for label in r.labels)


def test_train_val_test_split_covers_all_rows_without_overlap():
    rows, _ = build_multilabel_dataset(SAMPLE_CONDITIONS, synthetic_per_condition=10, seed=2)
    train, val, test = train_val_test_split(rows, seed=2)
    assert len(train) + len(val) + len(test) == len(rows)
