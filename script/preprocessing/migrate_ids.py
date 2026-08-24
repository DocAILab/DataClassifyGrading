"""Auditably migrate processed JSON records to the stable UUID5 identity.

Use only when the original tabular import is unavailable. Every field except
``id`` is preserved byte-for-value after JSON decoding; collisions fail before
any output is published.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from agent.hashing import sha256_file
from agent.task.identity import stable_record_id


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    ).encode("utf-8")


def _write_temp(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(content)
    except BaseException:
        Path(name).unlink(missing_ok=True)
        raise
    return Path(name)


def migrate_processed_ids(
    input_file: str | Path,
    output_file: str | Path,
    *,
    dataset: str,
    report_file: str | Path,
) -> dict[str, Any]:
    source = Path(input_file)
    output = Path(output_file)
    report_path = Path(report_file)
    if output.exists() or report_path.exists():
        raise FileExistsError("migration output and report must be new paths")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError("processed input must be a JSON array of objects")
    migrated = deepcopy(value)
    new_ids: list[str] = []
    changed = 0
    for index, row in enumerate(migrated):
        metadata = row.get("metadata")
        new_id = stable_record_id(dataset, metadata)
        if row.get("id") != new_id:
            changed += 1
        row["id"] = new_id
        new_ids.append(new_id)
    collisions = len(new_ids) - len(set(new_ids))
    if collisions:
        raise ValueError(
            f"stable identity collision count is {collisions}; migration aborted"
        )
    output_content = _json_bytes(migrated)
    report = {
        "format": "stable-record-id-migration-v1",
        "dataset": dataset,
        "records": len(migrated),
        "changed_ids": changed,
        "unchanged_ids": len(migrated) - changed,
        "identity_collisions": 0,
        "input_sha256": sha256_file(source),
        "output_sha256": hashlib.sha256(output_content).hexdigest(),
        "mutation_scope": ["id"],
    }
    output_temp = _write_temp(output, output_content)
    report_temp = _write_temp(report_path, _json_bytes(report))
    try:
        output_temp.replace(output)
        report_temp.replace(report_path)
    except BaseException:
        output_temp.unlink(missing_ok=True)
        report_temp.unlink(missing_ok=True)
        # Roll back either rename if the second publication failed.
        output.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
        raise
    return report


def _args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _args(argv)
    try:
        report = migrate_processed_ids(
            args.input,
            args.output,
            dataset=args.dataset,
            report_file=args.report,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"migrate_ids: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": "ok",
                "records": report["records"],
                "changed_ids": report["changed_ids"],
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
