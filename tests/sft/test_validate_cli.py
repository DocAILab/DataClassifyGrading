from __future__ import annotations

from script.verl.sft.validate import parse_args


def test_validate_cli_accepts_grading_config() -> None:
    args = parse_args(
        [
            "--dataset-dir",
            "dataset",
            "--registry",
            "registry.json",
            "--corpus",
            "corpus.json",
            "--metadata-fields",
            "field_name",
            "--grading-config",
            "grading.json",
        ]
    )
    assert args.grading_config == "grading.json"
