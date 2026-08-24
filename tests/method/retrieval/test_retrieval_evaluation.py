import json

import numpy as np
import pytest

from agent.task import LeafRegistry
from method.retrieval.evaluation import evaluate_stage1


class FakeEncoder:
    model_identity = "fake-bge"

    def encode_corpus(self, texts):
        vectors = np.eye(len(texts), dtype=np.float32)
        return vectors

    def encode_queries(self, texts):
        vectors = np.zeros((len(texts), 5), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, int(text.removeprefix("field"))] = 1.0
        return vectors


def _registry():
    return LeafRegistry.from_mapping({"categories": [
        {"category_id": label, "description": f"field{index}"}
        for index, label in enumerate("ABCDE")
    ]})


def test_evaluator_reads_only_val_and_writes_consistent_reports(tmp_path):
    rows = [
        {"id": "one", "metadata": {"field_name": "field0", "value": "secret"}, "classification": {"level_4": "A"}},
        {"id": "two", "metadata": {"field_name": "field1"}, "classification": {"level_4": "B"}},
    ]
    (tmp_path / "val.json").write_text(json.dumps(rows), encoding="utf-8")

    report = evaluate_stage1(tmp_path, _registry(), tmp_path / "out", encoder=FakeEncoder())

    assert report["requested_splits"] == ["val"]
    assert report["real_test_split_read"] is False
    assert report["metadata_fields"] == ["field_name"]
    assert report["rows"] == 2
    assert report["bge_m3"]["recall_at_1"] == 1.0
    assert report["runtime"]["gpu_peak_memory_bytes"] >= 0
    predictions = [json.loads(line) for line in (tmp_path / "out" / "bge_m3" / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(predictions) == 2
    assert set(predictions[0]) >= {"source_id", "field_name", "top5", "scores", "gold_rank"}
    assert "secret" not in json.dumps(predictions)
    assert (tmp_path / "out" / "COMPLETE").is_file()


def test_evaluator_rejects_duplicate_source_ids(tmp_path):
    row = {"id": "one", "metadata": {"field_name": "field0"}, "classification": {"level_4": "A"}}
    (tmp_path / "val.json").write_text(json.dumps([row, row]), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        evaluate_stage1(tmp_path, _registry(), tmp_path / "out", encoder=FakeEncoder())
