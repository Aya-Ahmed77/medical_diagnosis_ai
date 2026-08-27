"""
Shared tokenizer + padding utilities for the RNN and LSTM models.
Keeping this in one place guarantees both deep learning models use an
identical text -> integer sequence representation.
"""
import os
from typing import List

import joblib
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from app.config import get_config

TOKENIZER_FILENAME = "keras_tokenizer.joblib"


class SequenceVectorizer:
    def __init__(self):
        cfg = get_config()
        self.vocab_size = cfg.VOCAB_SIZE
        self.max_len = cfg.MAX_SEQUENCE_LENGTH
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token="<OOV>")
        self._fitted = False

    def fit(self, texts: List[str]) -> "SequenceVectorizer":
        self.tokenizer.fit_on_texts(texts)
        self._fitted = True
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("SequenceVectorizer must be fit() before transform().")
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len, padding="post", truncating="post")

    def save(self, local_dir: str) -> str:
        os.makedirs(local_dir, exist_ok=True)
        path = os.path.join(local_dir, TOKENIZER_FILENAME)
        joblib.dump(self.tokenizer, path)
        return path

    @classmethod
    def load(cls, local_dir: str) -> "SequenceVectorizer":
        vec = cls()
        path = os.path.join(local_dir, TOKENIZER_FILENAME)
        vec.tokenizer = joblib.load(path)
        vec._fitted = True
        return vec
