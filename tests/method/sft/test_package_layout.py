from pathlib import Path


def test_sft_lives_under_lowercase_method_package() -> None:
    root = Path(__file__).resolve().parents[3]
    assert (root / "src/method/sft/__init__.py").is_file()
    assert not (root / "src/algorithm").exists()
