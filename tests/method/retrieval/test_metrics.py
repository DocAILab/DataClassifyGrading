import pytest

from method.retrieval.metrics import summarize_retrieval


def _row(source_id, gold, ranked):
    return {"source_id": source_id, "golden_level_4": gold, "ranked_labels": ranked}


def test_metrics_compute_recall_macro_mrr_rank_and_coverage():
    rows = [
        _row("1", "A", ["A", "B", "C"]),
        _row("2", "B", ["A", "B", "C"]),
    ]

    report = summarize_retrieval(rows, registry_ids=("A", "B", "C"), train_counts={"A": 3, "B": 1, "C": 0})

    assert report["recall_at_1"] == pytest.approx(0.5)
    assert report["recall_at_3"] == 1.0
    assert report["macro_recall_at_1"] == pytest.approx(0.5)
    assert report["macro_class_count"] == 2
    assert report["mrr"] == pytest.approx(0.75)
    assert report["mean_gold_rank"] == pytest.approx(1.5)
    assert report["registry_top5_coverage_count"] == 3
    assert report["unique_top5_tuples"] == 1


def test_metrics_reject_duplicate_source_ids_and_oov_gold():
    with pytest.raises(ValueError, match="duplicate source_id"):
        summarize_retrieval([_row("1", "A", ["A"]), _row("1", "A", ["A"])], registry_ids=("A",), train_counts={})
    with pytest.raises(ValueError, match="gold"):
        summarize_retrieval([_row("1", "Z", ["A"])], registry_ids=("A",), train_counts={})
