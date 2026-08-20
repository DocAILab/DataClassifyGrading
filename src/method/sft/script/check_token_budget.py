"""Check exported SFT message lengths with a model's real chat template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from method.sft import inspect_token_budget


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--model", required=True, help="Local path or Hugging Face model ID")
    parser.add_argument("--max-length", type=int, required=True)
    parser.add_argument("--report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        report = inspect_token_budget(
            args.dataset_dir,
            tokenizer,
            max_length=args.max_length,
        )
        rendered = json.dumps(report, ensure_ascii=False, indent=2)
        if args.report:
            path = Path(args.report)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(rendered, encoding="utf-8")
        print(rendered)
        return 0 if report["valid"] else 1
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"check_token_budget: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
