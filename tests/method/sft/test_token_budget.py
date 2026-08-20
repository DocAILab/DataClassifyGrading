from pathlib import Path

from method.sft import export_sft_dataset, inspect_token_budget


class LengthTokenizer:
    def apply_chat_template(self, messages, **_kwargs):
        return list(range(sum(len(message["content"]) for message in messages)))


def test_token_budget_reports_longest_rows_and_limit_violations(tmp_path: Path):
    fixture_dir = Path(__file__).parent / "fixtures"
    output_dir = tmp_path / "sft"
    export_sft_dataset(
        fixture_dir,
        output_dir,
        fixture_dir / "registry.json",
        fixture_dir / "task.json",
    )

    baseline = inspect_token_budget(output_dir, LengthTokenizer(), max_length=10_000)
    constrained = inspect_token_budget(output_dir, LengthTokenizer(), max_length=10)

    assert baseline["valid"] is True
    assert baseline["splits"]["train"]["rows"] == 2
    assert baseline["splits"]["train"]["max_tokens"] > 0
    assert baseline["splits"]["train"]["longest_source_id"] == "fixture-1"
    assert constrained["valid"] is False
    assert constrained["splits"]["train"]["over_limit"] == 2


def test_token_budget_accepts_explicit_train_val_without_test(tmp_path: Path):
    fixture_dir = Path(__file__).parent / "fixtures"
    output_dir = tmp_path / "sft"
    export_sft_dataset(
        fixture_dir,
        output_dir,
        fixture_dir / "registry.json",
        fixture_dir / "task.json",
        splits=("train", "val"),
    )

    report = inspect_token_budget(
        output_dir,
        LengthTokenizer(),
        max_length=10_000,
        splits=("train", "val"),
    )

    assert report["valid"] is True
    assert set(report["splits"]) == {"train", "val"}
    assert report["real_test_split_read"] is False
