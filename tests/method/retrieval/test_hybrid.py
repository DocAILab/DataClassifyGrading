import numpy as np

from method.retrieval.hybrid import (
    build_field_label_index,
    field_index_scores,
    fuse_retrieval_scores,
)


def test_field_index_uses_train_only_exact_field_label_counts():
    registry = ("A", "B", "C")
    train = [
        {"metadata": {"field_name": "order_id"}, "classification": {"level_4": "B"}},
        {"metadata": {"field_name": "order_id"}, "classification": {"level_4": "B"}},
        {"metadata": {"field_name": "order_id"}, "classification": {"level_4": "A"}},
    ]
    index = build_field_label_index(train, registry)
    scores = field_index_scores([" order_id "], index, registry)
    assert scores.shape == (1, 3)
    assert np.argmax(scores[0]) == 1
    assert scores[0, 1] > scores[0, 0] > scores[0, 2]


def test_field_index_nearest_seen_field_handles_unseen_query_without_gold_access():
    registry = ("A", "B", "C")
    train = [
        {"metadata": {"field_name": "customer_phone"}, "classification": {"level_4": "B"}},
        {"metadata": {"field_name": "customer_name"}, "classification": {"level_4": "A"}},
    ]
    index = build_field_label_index(train, registry)
    scores = field_index_scores(["customer_phone_ext"], index, registry)
    assert np.argmax(scores[0]) == 1


def test_fusion_is_row_normalized_and_keeps_deterministic_shape():
    lexical = np.array([[0.2, 0.4, 0.1]])
    dense = np.array([[0.8, 0.1, 0.3]])
    index = np.array([[0.0, 2.0, 1.0]])
    fused = fuse_retrieval_scores(
        lexical,
        dense,
        index,
        lexical_weight=0.2,
        dense_weight=0.3,
        index_weight=0.5,
    )
    assert fused.shape == lexical.shape
    assert np.isfinite(fused).all()
    assert np.argmax(fused[0]) == 1
