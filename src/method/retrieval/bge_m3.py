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
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.batch_size = batch_size
        self.device = device
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True
        )
        dtype = torch.float16 if use_fp16 and device.startswith("cuda") else torch.float32
        self._model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True, torch_dtype=dtype
        ).to(device).eval()

    def _encode(self, texts: Sequence[str], *, name: str) -> np.ndarray:
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            chunk = list(texts[start:start + self.batch_size])
            encoded = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with self._torch.inference_mode():
                hidden = self._model(**encoded, return_dict=True).last_hidden_state[:, 0]
                hidden = self._torch.nn.functional.normalize(hidden, p=2, dim=-1)
            vectors.append(hidden.float().cpu().numpy())
        if not vectors:
            raise ValueError(f"{name} texts must not be empty")
        return _validated_vectors(np.concatenate(vectors, axis=0), name=name)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, name="query")

    def encode_corpus(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, name="corpus")
