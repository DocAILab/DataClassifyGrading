"""Lazy BGE-M3 dense encoder and validated similarity scoring."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np


def _validated_vectors(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} vectors must be a finite rank-2 array")
    norms = np.linalg.norm(array, axis=1)
    if not np.allclose(norms, 1.0, atol=2e-3):
        raise ValueError(f"{name} vectors must be normalized")
    return array


def dense_scores(query_vectors, corpus_vectors) -> np.ndarray:
    queries = _validated_vectors(query_vectors, name="query")
    corpus = _validated_vectors(corpus_vectors, name="corpus")
    if queries.shape[1] != corpus.shape[1]:
        raise ValueError("query and corpus embedding dimensions must match")
    return queries @ corpus.T


class BgeM3DenseEncoder:
    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        batch_size: int = 32,
        use_fp16: bool = True,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.batch_size = batch_size
        self._model = BGEM3FlagModel(
            str(model_path), devices=device, pooling_method="cls", use_fp16=use_fp16
        )

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        output = self._model.encode_queries(
            list(texts), batch_size=self.batch_size,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        return _validated_vectors(output["dense_vecs"], name="query")

    def encode_corpus(self, texts: Sequence[str]) -> np.ndarray:
        output = self._model.encode_corpus(
            list(texts), batch_size=self.batch_size,
            return_dense=True, return_sparse=False, return_colbert_vecs=False,
        )
        return _validated_vectors(output["dense_vecs"], name="corpus")
