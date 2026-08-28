"""Evaluate frozen BGE-M3 and character n-gram Stage 1 retrieval on val only."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence

from agent.task import LeafRegistry
from method.retrieval.bge_m3 import BgeM3DenseEncoder
from method.retrieval.evaluation import evaluate_stage1


def model_identity(model_path: Path) -> str:
    digest = sha256()
    files = [model_path / "config.json"]
    weights = sorted(model_path.glob("*.safetensors")) or sorted(model_path.glob("pytorch_model*.bin"))
    if not weights or not files[0].is_file():
        raise FileNotFoundError("BGE-M3 config and model weights are required")
    for path in [*files, *weights]:
        digest.update(path.name.encode("utf-8") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lexical-weight", type=float, default=0.20)
    parser.add_argument("--dense-weight", type=float, default=0.50)
    parser.add_argument("--index-weight", type=float, default=0.30)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    encoder = BgeM3DenseEncoder(args.model, batch_size=args.batch_size)
    encoder.model_identity = model_identity(args.model)
    report = evaluate_stage1(
        args.input_dir,
        LeafRegistry.from_path(args.registry),
        args.output_dir,
        encoder=encoder,
        limit=args.limit,
        lexical_weight=args.lexical_weight,
        dense_weight=args.dense_weight,
        index_weight=args.index_weight,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
