"""Character n-gram lexical control using the established DPO scorer."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from method.dpo.label_scoring import _cosine, _features


def char_ngram_scores(queries: Sequence[str], documents: Sequence[str]) -> np.ndarray:
    query_features = [_features(value) for value in queries]
    document_features = [_features(value) for value in documents]
    return np.asarray(
        [[_cosine(query, document) for document in document_features] for query in query_features],
        dtype=np.float64,
    )
