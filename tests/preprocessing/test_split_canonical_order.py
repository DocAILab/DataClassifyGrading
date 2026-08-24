"""Row-order insensitivity and report-anchor tests for the splitter."""

from __future__ import annotations

import json
import random
from pathlib import Path

from script.preprocessing.split import split_dataset


def _records(count: int = 60) -> list[dict]:
    records = []
    for index in range(count):
        records.append(
            {
                "id": f"row-{index:03d}",
                "metadata": {"table_name": f"T{index % 7}"},
                "classification": {
                    "level_1": f"L1-{index % 3}",
                    "level_4": f"leaf-{index % 5}",
                },
                "data_level": "L2",
            }
        )
    return records


def _write(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    return path


def test_split_ignores_input_row_order(tmp_path) -> None:
    records = _records()
    shuffled = records.copy()
    random.Random(7).shuffle(shuffled)

    first_dir = tmp_path / "ordered"
    second_dir = tmp_path / "shuffled"
    split_dataset(_write(tmp_path / "in-a.json", records), first_dir, overwrite=True)
    split_dataset(_write(tmp_path / "in-b.json", shuffled), second_dir, overwrite=True)

    for name in ("train.json", "val.json", "test.json", "split_report.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_report_carries_reproducibility_anchors(tmp_path) -> None:
    out = tmp_path / "out"
    split_dataset(
        _write(tmp_path / "in.json", _records()),
        out,
        split_type="group",
        group_key="metadata.table_name",
        random_seed=99,
        overwrite=True,
    )
    report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
    assert report["seed"] == 99
    assert report["order_rule"] == "id-ascending"
    assert report["split_type"] == "group"
    assert report["group_key"] == "metadata.table_name"
    assert report["ratios"] == {"train": 0.8, "val": 0.1, "test": 0.1}
    # same seed + same data => identical membership
    again = tmp_path / "again"
    split_dataset(
        _write(tmp_path / "in2.json", list(reversed(_records()))),
        again,
        split_type="group",
        group_key="metadata.table_name",
        random_seed=99,
        overwrite=True,
    )
    assert (again / "train.json").read_bytes() == (out / "train.json").read_bytes()
