from pathlib import Path

import pytest

from agent.task import ClassificationAssets


ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / "cfg" / "task"


def test_synthetic_runtime_assets_load_through_public_interface() -> None:
    assets = ClassificationAssets.from_files(
        registry=TASK_DIR / "leaf_registry.example.json",
        corpus=TASK_DIR / "corpus.example.json",
        task=TASK_DIR / "task.example.json",
    )

    assert len(assets.registry.categories) == 6
    assert assets.task.metadata_fields == ("title", "summary")
    assert assets.corpus is not None
    assert set(assets.corpus) == set(assets.registry.ids)


def test_runtime_assets_reject_corpus_ids_outside_registry(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        '{"categories":[{"category_id":"outside","name":"Outside"}]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absent from the leaf registry"):
        ClassificationAssets.from_files(
            registry=TASK_DIR / "leaf_registry.example.json",
            corpus=corpus,
            task=TASK_DIR / "task.example.json",
        )


def test_runtime_corpus_is_optional() -> None:
    assets = ClassificationAssets.from_files(
        registry=TASK_DIR / "leaf_registry.example.json",
        task=TASK_DIR / "task.example.json",
    )
    assert assets.corpus is None
