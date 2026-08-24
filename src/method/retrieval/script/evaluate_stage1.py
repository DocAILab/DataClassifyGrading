"""Evaluate frozen BGE-M3 and character n-gram Stage 1 retrieval on val only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from agent.task import LeafRegistry
from method.retrieval.bge_m3 import BgeM3DenseEncoder
from method.retrieval.evaluation import evaluate_stage1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = BgeM3DenseEncoder(args.model, batch_size=args.batch_size)
    encoder.model_identity = f"local:{args.model.resolve()}"
    report = evaluate_stage1(
        args.input_dir,
        LeafRegistry.from_path(args.registry),
        args.output_dir,
        encoder=encoder,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
