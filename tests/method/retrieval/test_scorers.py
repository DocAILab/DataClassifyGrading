import numpy as np
import pytest

from method.retrieval.bge_m3 import dense_scores
from method.retrieval.char_ngram import char_ngram_scores


def test_dense_scores_require_normalized_aligned_vectors():
    queries = np.array([[1.0, 0.0]])
    corpus = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert dense_scores(queries, corpus).tolist() == [[1.0, 0.0]]
    with pytest.raises(ValueError, match="normalized"):
        dense_scores(queries * 2, corpus)


def test_char_ngram_prefers_overlapping_chinese_label_text():
    scores = char_ngram_scores(["设备编号"], ["标签名称：设备基本资料", "标签名称：车辆信息"])
    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]
