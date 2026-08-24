"""Fingerprint a trained SFT checkpoint for reference-checkpoint lineage.

Produces a deterministic SHA-256 "tree hash" over every file in the
checkpoint directory (sorted relative paths, path+size+content digested),
so all RL algorithms can verify they start from THE shared reference
checkpoint. Optionally links the export_report.json that produced the
training parquet, closing the provenance chain:

    export_report.json (parquet sha256) -> checkpoint_sha256

Example:
    python -m script.verl.sft.record_checkpoint \\
        --checkpoint-dir <local>/checkpoints/reference-<dataset> \\
        --export-report <local>/sft/<dataset>/export_report.json \\
        --output <local>/checkpoints/reference-<dataset>.provenance.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from agent.hashing import sha256_file

CHECKPOINT_HASH_ALGORITHM = "sha256-tree-v1"


def tree_hash(root: Path) -> tuple[str, int, int]:
    """Deterministic digest over all files under *root* (recursive).

    Returns ``(digest, file_count, total_bytes)``. File contents are read in
    chunks; directory layout only contributes via each file's POSIX-relative
    path and size.
    """
    if not root.is_dir():
        raise NotADirectoryError(f"checkpoint directory not found: {root}")
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {root}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0".encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        total_bytes += size
    return digest.hexdigest(), len(files), total_bytes


def build_provenance(
    checkpoint_dir: str | Path,
    export_report: str | Path | None = None,
) -> dict:
    """Build the provenance mapping (pure computation, nothing written)."""
    root = Path(checkpoint_dir)
    fingerprint, file_count, total_bytes = tree_hash(root)
    provenance = {
        "algorithm": CHECKPOINT_HASH_ALGORITHM,
        "checkpoint_dir": root.as_posix(),
        "checkpoint_sha256": fingerprint,
        "files": file_count,
        "total_bytes": total_bytes,
    }
    if export_report is not None:
        report_path = Path(export_report)
        if not report_path.is_file():
            raise FileNotFoundError(f"export report not found: {report_path}")
        provenance["training_export_report"] = {
            "path": report_path.as_posix(),
            "sha256": sha256_file(report_path),
        }
    return provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Trained reference checkpoint directory")
    parser.add_argument("--export-report",
                        help="Optional export_report.json linking training parquet sha256")
    parser.add_argument("--output", required=True,
                        help="Destination provenance JSON (atomic write)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"output already exists: {output} (pass --overwrite)", file=sys.stderr)
        return 2
    provenance = build_provenance(args.checkpoint_dir, args.export_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(provenance, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(json.dumps({"checkpoint_sha256": provenance["checkpoint_sha256"],
                      "files": provenance["files"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
