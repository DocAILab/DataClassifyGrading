"""Release and verify the merged HF model used as the unique RL reference.

A reference release is accepted only when its complete lineage verifies:

    passed/approved-waived SFT export + actual parquet hashes
      -> effective VeRL config + environment + base model
      -> merged HF model tree hash

The generated provenance is deterministic (no timestamps).  RLOO launchers
must call :func:`verify_reference_provenance` against the exact model
folder they load.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from agent.hashing import sha256_file
from agent.training.input_audit import (
    require_identifiable_prompt_bundle,
    require_identifiable_prompts,
)

CHECKPOINT_HASH_ALGORITHM = "sha256-tree-v2"
_ARTIFACT_KIND = "merged_hf_reference_model"
_SPLITS = ("train", "val", "test")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")


def tree_hash(root: str | Path) -> tuple[str, int, int]:
    """Return a deterministic digest, file count and byte count for *root*."""
    directory = Path(root)
    if not directory.is_dir():
        raise NotADirectoryError(f"checkpoint directory not found: {directory}")
    files = sorted(path for path in directory.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"checkpoint directory contains no files: {directory}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(directory).as_posix()
        size = path.stat().st_size
        digest.update(f"{relative}\0{size}\0".encode("utf-8"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        total_bytes += size
    return digest.hexdigest(), len(files), total_bytes


def _json_object(path: Path, description: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a JSON object: {path}")
    return value


def _validate_hf_model_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{description} directory not found: {path}")
    if not (path / "config.json").is_file():
        raise ValueError(f"{description} has no config.json: {path}")
    weights = [*path.glob("*.safetensors"), *path.glob("pytorch_model*.bin")]
    if not weights:
        raise ValueError(f"{description} has no HF model weights: {path}")
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "spiece.model",
    )
    if not any((path / name).is_file() for name in tokenizer_files):
        raise ValueError(f"{description} has no tokenizer artifact: {path}")


def _resolve_report_artifact(report: Path, raw_path: object) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("export report split output_file must be a non-empty path")
    supplied = Path(raw_path)
    candidates = [supplied] if supplied.is_absolute() else [report.parent / supplied, supplied]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"export report parquet not found: {raw_path!r} "
        f"(resolved relative to {report.parent})"
    )


def _verify_export_report(path: Path) -> dict[str, Any]:
    report = _json_object(path, "SFT export report")
    release = report.get("release")
    if (
        not isinstance(release, Mapping)
        or release.get("status") != "passed"
        or release.get("published") is not True
    ):
        raise ValueError(
            "export report must describe a passed published release status"
        )
    gate = report.get("label_gap_gate")
    if not isinstance(gate, Mapping):
        raise ValueError("export report has no label_gap_gate object")
    status = gate.get("status")
    if status not in {"passed", "waived"}:
        raise ValueError(
            f"export report gate status must be passed or waived, got {status!r}"
        )
    gap_fields = ("blocking", "blocking_levels", "waived", "waived_levels")
    gap_values: dict[str, list[Any]] = {}
    for field in gap_fields:
        raw = gate.get(field, [])
        if raw is None:
            raw = []
        if not isinstance(raw, list):
            raise ValueError(f"export report label_gap_gate.{field} must be an array")
        gap_values[field] = raw
    if gap_values["blocking"] or gap_values["blocking_levels"]:
        raise ValueError("export report still contains blocking gaps")
    if status == "passed" and (
        gap_values["waived"] or gap_values["waived_levels"]
    ):
        raise ValueError("passed export report must not contain waived gaps")

    report_format = report.get("format")
    if report_format != "dataclassify-finance-shougang-mixture-v1":
        raise ValueError("reference checkpoint requires a finance+shougang SFT mixture report")
    if report.get("family") != "sft":
        raise ValueError("reference checkpoint requires an SFT mixture report")
    raw_inputs = report.get("inputs")
    input_datasets = (
        sorted(str(dataset) for dataset in raw_inputs)
        if isinstance(raw_inputs, Mapping)
        else []
    )
    is_joint_mixture = (
        report_format == "dataclassify-finance-shougang-mixture-v1"
        or {"finance", "shougang"}.issubset(set(input_datasets))
    )
    if set(input_datasets) != {"finance", "shougang"}:
        raise ValueError("SFT mixture report inputs must be exactly finance and shougang")
    if is_joint_mixture and (
        status != "passed" or gap_values["waived"] or gap_values["waived_levels"]
    ):
        raise ValueError("joint finance+shougang release cannot contain gap waivers")

    raw_splits = report.get("splits")
    if not isinstance(raw_splits, Mapping):
        raise ValueError("export report has no splits object")
    parquet_sha256: dict[str, str] = {}
    parquet_paths: dict[str, str] = {}
    for split in _SPLITS:
        split_report = raw_splits.get(split)
        if not isinstance(split_report, Mapping):
            raise ValueError(f"export report has no {split!r} split")
        parquet = _resolve_report_artifact(path, split_report.get("output_file"))
        expected = split_report.get("parquet_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"export report {split} parquet_sha256 is invalid")
        actual = sha256_file(parquet)
        if actual != expected:
            raise ValueError(
                f"parquet sha256 mismatch for {split}: expected {expected}, got {actual}"
            )
        parquet_sha256[split] = actual
        parquet_paths[split] = parquet.as_posix()
    lineage: dict[str, Any] = {
        "path": path.as_posix(),
        "sha256": sha256_file(path),
        "gate_status": status,
        "parquet_paths": parquet_paths,
        "parquet_sha256": parquet_sha256,
        "format": report_format,
        "family": report.get("family"),
        "datasets": input_datasets,
    }
    raw_manifest = report.get("grading_manifest")
    if not isinstance(raw_manifest, Mapping):
        raise ValueError("SFT mixture report must contain grading_manifest lineage")
    manifest_path = raw_manifest.get("path")
    manifest_sha = raw_manifest.get("sha256")
    if not isinstance(manifest_path, str) or not isinstance(manifest_sha, str):
        raise ValueError("SFT mixture grading_manifest lineage is malformed")
    manifest_source = Path(manifest_path)
    if not manifest_source.is_absolute():
        manifest_source = path.parent / manifest_source
    manifest_lineage = _verify_grading_manifest(manifest_source)
    if manifest_lineage["sha256"] != manifest_sha:
        raise ValueError("SFT mixture grading_manifest sha256 mismatch")
    if raw_manifest.get("datasets") != manifest_lineage["datasets"]:
        raise ValueError("SFT mixture grading_manifest dataset lineage mismatch")
    lineage["grading_manifest"] = manifest_lineage
    return lineage


def _verify_prompt_audit_file(
    path: str | Path,
    *,
    require_bundle: bool,
    expected_datasets: set[str] | None = None,
    expected_dataset: str | None = None,
    expected_grading_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    source = Path(path)
    report = _json_object(source, "prompt-identifiability report")
    report_format = report.get("format")
    if report_format == "field-prompt-identifiability-bundle-v1":
        try:
            require_identifiable_prompt_bundle(report)
        except ValueError as exc:
            raise ValueError(f"prompt-identifiability bundle rejected: {exc}") from exc
        datasets = report["datasets"]
        actual_datasets = set(datasets)
        if expected_datasets is not None and actual_datasets != expected_datasets:
            raise ValueError("prompt-identifiability bundle datasets do not match release")
        if expected_grading_hashes is not None:
            for dataset, expected in expected_grading_hashes.items():
                standard = report["standard_hashes"].get(dataset)
                if not isinstance(standard, Mapping) or standard.get(
                    "grading_standard_sha256"
                ) != expected:
                    raise ValueError(
                        f"prompt-identifiability {dataset} grading standard mismatch"
                    )
        return {
            "kind": "bundle",
            "path": source.as_posix(),
            "sha256": sha256_file(source),
            "status": "passed",
            "conflict_keys": 0,
            "datasets": sorted(actual_datasets),
            "standard_hashes": report["standard_hashes"],
        }
    if require_bundle:
        raise ValueError(
            "prompt-identifiability: joint finance+shougang release requires per-dataset prompt-audit bundle"
        )
    if expected_dataset is not None and report.get("dataset") != expected_dataset:
        raise ValueError(
            f"prompt-identifiability report dataset does not match {expected_dataset}"
        )
    try:
        require_identifiable_prompts(report)
    except ValueError as exc:
        raise ValueError(f"prompt-identifiability report rejected: {exc}") from exc
    return {
        "kind": "single",
        "path": source.as_posix(),
        "sha256": sha256_file(source),
        "status": "passed",
        "conflict_keys": 0,
        "classification_standard_sha256": report.get(
            "classification_standard_sha256"
        ),
        "grading_standard_sha256": report.get("grading_standard_sha256"),
        "dataset": report.get("dataset"),
    }


def _verify_prompt_audit(
    value: str | Path | Mapping[str, str] | Sequence[str | Path],
    *,
    require_bundle: bool,
    expected_datasets: set[str] | None = None,
    expected_grading_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Verify a single report or a dataset-local report mapping.

    A joint release must use a serialized bundle.  Mapping/sequence support is
    retained for callers that already hold two report paths; it is normalized
    to the same redacted per-dataset lineage shape but is not accepted as a
    substitute for a bundle when ``require_bundle`` is true.
    """

    if isinstance(value, Mapping):
        raise ValueError(
            "prompt-identifiability report mapping is not a serialized bundle"
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, Path)):
        paths = list(value)
        if len(paths) != 2:
            raise ValueError("prompt-audit report sequence must contain both datasets")
        return _verify_prompt_audit(
            {dataset: paths[index] for index, dataset in enumerate(("finance", "shougang"))},
            require_bundle=require_bundle,
            expected_datasets=expected_datasets,
            expected_grading_hashes=expected_grading_hashes,
        )
    return _verify_prompt_audit_file(
        value,
        require_bundle=require_bundle,
        expected_datasets=expected_datasets,
        expected_grading_hashes=expected_grading_hashes,
    )


def _validate_effective_config(path: str | Path) -> Path:
    """Require a non-empty JSON/YAML mapping, not an arbitrary byte artifact."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"effective config not found: {source}")
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"effective config is not valid UTF-8: {source}") from exc
    if not text.strip() or "\x00" in text:
        raise ValueError("effective config must be a non-empty mapping")
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # PyYAML is optional.  Prefer it when present, then use a conservative
        # root-mapping check so this release gate does not depend on it.
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            yaml = None
        if yaml is not None:
            try:
                parsed = yaml.safe_load(text)
            except Exception as exc:
                raise ValueError("effective config is not valid YAML/JSON") from exc
        else:
            top_level_keys = []
            for line in text.splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                if line[0].isspace():
                    continue
                if ":" not in line:
                    raise ValueError("effective config must be a YAML/JSON mapping")
                key, _ = line.split(":", 1)
                key = key.strip()
                if not key or key.startswith(("-", "[", "{")):
                    raise ValueError("effective config must be a YAML/JSON mapping")
                top_level_keys.append(key)
            if not top_level_keys or len(set(top_level_keys)) != len(top_level_keys):
                raise ValueError("effective config must be a non-empty mapping")
            parsed = {key: True for key in top_level_keys}
    if not isinstance(parsed, Mapping) or not parsed:
        raise ValueError("effective config must be a non-empty mapping")
    return source


def _file_lineage(path: str | Path, description: str) -> dict[str, str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{description} not found: {source}")
    return {"path": source.as_posix(), "sha256": sha256_file(source)}


def _verify_grading_manifest(path: str | Path) -> dict[str, Any]:
    """Load and hash the dataset-local grading standards manifest."""

    from agent.task.grading_manifest import DatasetGradingManifest

    source = Path(path)
    manifest = DatasetGradingManifest.from_path(source)
    return {
        "path": source.as_posix(),
        "sha256": sha256_file(source),
        "datasets": {
            dataset: {"sha256": manifest.sha256_for(dataset)}
            for dataset in ("finance", "shougang")
        },
    }


def build_provenance(
    checkpoint_dir: str | Path,
    export_report: str | Path,
    *,
    effective_config: str | Path,
    base_model: str | Path,
    environment_report: str | Path,
    prompt_audit_report: (
        str | Path | Mapping[str, str] | Sequence[str | Path]
    ),
    git_commit: str,
    global_step: int,
    grading_manifest: str | Path,
) -> dict[str, Any]:
    """Verify all release inputs and return deterministic reference lineage."""
    if not _GIT_COMMIT_RE.fullmatch(git_commit):
        raise ValueError("git commit must be a 7-40 character hexadecimal commit id")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise ValueError("global step must be a positive integer")

    checkpoint = Path(checkpoint_dir)
    base = Path(base_model)
    _validate_hf_model_dir(checkpoint, "merged HF checkpoint")
    _validate_hf_model_dir(base, "base HF model")
    # Requiring parseable environment JSON prevents a random file from being
    # used as the only environment anchor.
    environment_path = Path(environment_report)
    _json_object(environment_path, "environment report")

    training_lineage = _verify_export_report(Path(export_report))
    joint_release = set(training_lineage.get("datasets", [])) == {
        "finance", "shougang"
    }
    manifest_lineage = _verify_grading_manifest(grading_manifest)
    expected_grading_hashes: dict[str, str] = {}
    if manifest_lineage is not None:
        expected_grading_hashes = {
            dataset: details["sha256"]
            for dataset, details in manifest_lineage["datasets"].items()
        }
    else:
        # A mixture may carry a manifest lineage from its source release.  If
        # present, use it as an additional binding for the prompt audit.
        raw_manifest = training_lineage.get("grading_manifest")
        if isinstance(raw_manifest, Mapping):
            raw_datasets = raw_manifest.get("datasets")
            if isinstance(raw_datasets, Mapping):
                for dataset, details in raw_datasets.items():
                    if isinstance(details, Mapping) and isinstance(
                        details.get("sha256"), str
                    ):
                        expected_grading_hashes[str(dataset)] = details["sha256"]

    checkpoint_sha, files, total_bytes = tree_hash(checkpoint)
    base_sha, base_files, base_bytes = tree_hash(base)
    prompt_lineage = _verify_prompt_audit(
        prompt_audit_report,
        require_bundle=joint_release,
        expected_datasets={"finance", "shougang"} if joint_release else None,
        expected_grading_hashes=expected_grading_hashes or None,
    )
    provenance: dict[str, Any] = {
        "algorithm": CHECKPOINT_HASH_ALGORITHM,
        "artifact_kind": _ARTIFACT_KIND,
        "checkpoint_dir": checkpoint.as_posix(),
        "checkpoint_sha256": checkpoint_sha,
        "files": files,
        "total_bytes": total_bytes,
        "global_step": global_step,
        "git_commit": git_commit.lower(),
        "training_export_report": training_lineage,
        "effective_config": _file_lineage(effective_config, "effective config"),
        "environment_report": _file_lineage(environment_path, "environment report"),
        "prompt_identifiability": prompt_lineage,
        "base_model": {
            "path": base.as_posix(),
            "algorithm": CHECKPOINT_HASH_ALGORITHM,
            "sha256": base_sha,
            "files": base_files,
            "total_bytes": base_bytes,
        },
    }
    if manifest_lineage is not None:
        provenance["grading_manifest"] = manifest_lineage
    return provenance


def verify_reference_provenance(
    provenance_path: str | Path,
    model_dir: str | Path,
) -> Mapping[str, Any]:
    """Fail closed unless the complete released reference lineage verifies."""

    path = Path(provenance_path)
    provenance = _json_object(path, "reference provenance")
    required = {
        "algorithm", "artifact_kind", "checkpoint_dir", "checkpoint_sha256",
        "files", "total_bytes", "global_step", "git_commit",
        "training_export_report", "effective_config", "environment_report",
        "prompt_identifiability", "base_model", "grading_manifest",
    }
    if set(provenance) != required:
        raise ValueError("reference provenance is missing required lineage fields")
    if provenance.get("algorithm") != CHECKPOINT_HASH_ALGORITHM:
        raise ValueError("reference provenance uses an unsupported hash algorithm")
    if provenance.get("artifact_kind") != _ARTIFACT_KIND:
        raise ValueError("reference provenance is not a merged HF reference model")
    git_commit = provenance.get("git_commit")
    if not isinstance(git_commit, str) or not _GIT_COMMIT_RE.fullmatch(git_commit):
        raise ValueError("reference provenance git commit is invalid")
    global_step = provenance.get("global_step")
    if isinstance(global_step, bool) or not isinstance(global_step, int) or global_step <= 0:
        raise ValueError("reference provenance global step is invalid")

    model = Path(model_dir)
    _validate_hf_model_dir(model, "reference HF model")
    if provenance.get("checkpoint_dir") != model.as_posix():
        raise ValueError("reference provenance checkpoint path does not match model")
    actual, files, total_bytes = tree_hash(model)
    expected = provenance.get("checkpoint_sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("reference provenance checkpoint sha256 is invalid")
    if actual != expected:
        raise ValueError(
            f"checkpoint sha256 mismatch: expected {expected!r}, got {actual}"
        )
    if provenance.get("files") != files or provenance.get("total_bytes") != total_bytes:
        raise ValueError("checkpoint file-count/byte-count provenance mismatch")

    training = provenance["training_export_report"]
    if not isinstance(training, Mapping):
        raise ValueError("reference provenance training report lineage is missing")
    report_path = training.get("path")
    expected_report_sha = training.get("sha256")
    if not isinstance(report_path, str) or not isinstance(expected_report_sha, str):
        raise ValueError("reference provenance training report lineage is malformed")
    report_file = Path(report_path)
    if sha256_file(report_file) != expected_report_sha:
        raise ValueError("training export report sha256 mismatch")
    verified_training = _verify_export_report(report_file)
    for field in ("parquet_sha256", "gate_status", "format", "family", "datasets"):
        if verified_training.get(field) != training.get(field):
            raise ValueError(f"training export report {field} lineage mismatch")

    def verify_file_lineage(value: object, description: str) -> Path:
        if not isinstance(value, Mapping):
            raise ValueError(f"reference provenance {description} lineage is missing")
        raw_path = value.get("path")
        expected_sha = value.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_sha, str):
            raise ValueError(f"reference provenance {description} lineage is malformed")
        source = Path(raw_path)
        if sha256_file(source) != expected_sha:
            raise ValueError(f"{description} sha256 mismatch")
        return source

    effective = verify_file_lineage(provenance["effective_config"], "effective config")
    environment = verify_file_lineage(provenance["environment_report"], "environment report")
    _json_object(environment, "environment report")

    base_lineage = provenance["base_model"]
    if not isinstance(base_lineage, Mapping):
        raise ValueError("reference provenance base model lineage is missing")
    base_path = base_lineage.get("path")
    if not isinstance(base_path, str) or base_lineage.get("algorithm") != CHECKPOINT_HASH_ALGORITHM:
        raise ValueError("reference provenance base model lineage is malformed")
    base = Path(base_path)
    _validate_hf_model_dir(base, "base HF model")
    base_actual, base_files, base_bytes = tree_hash(base)
    if (
        base_lineage.get("sha256") != base_actual
        or base_lineage.get("files") != base_files
        or base_lineage.get("total_bytes") != base_bytes
    ):
        raise ValueError("base model provenance mismatch")

    grading_lineage = provenance["grading_manifest"]
    if not isinstance(grading_lineage, Mapping):
        raise ValueError("reference provenance grading manifest lineage is missing")
    manifest_path = grading_lineage.get("path")
    expected_manifest_sha = grading_lineage.get("sha256")
    if not isinstance(manifest_path, str) or not isinstance(expected_manifest_sha, str):
        raise ValueError("reference provenance grading manifest lineage is malformed")
    manifest_file = Path(manifest_path)
    if sha256_file(manifest_file) != expected_manifest_sha:
        raise ValueError("grading manifest sha256 mismatch")
    verified_manifest = _verify_grading_manifest(manifest_file)
    if verified_manifest["datasets"] != grading_lineage.get("datasets"):
        raise ValueError("grading manifest dataset lineage mismatch")

    prompt = provenance["prompt_identifiability"]
    if not isinstance(prompt, Mapping):
        raise ValueError("reference provenance requires prompt-identifiability lineage")
    expected_grading_hashes = {
        dataset: details["sha256"]
        for dataset, details in verified_manifest["datasets"].items()
    }
    if prompt.get("kind") == "bundle":
        prompt_path = prompt.get("path")
        expected_prompt_sha = prompt.get("sha256")
        if not isinstance(prompt_path, str) or not isinstance(expected_prompt_sha, str):
            raise ValueError("reference provenance prompt bundle lineage is malformed")
        prompt_file = Path(prompt_path)
        if sha256_file(prompt_file) != expected_prompt_sha:
            raise ValueError("prompt-identifiability bundle sha256 mismatch")
        try:
            verified_prompt = _verify_prompt_audit_file(
                prompt_file,
                require_bundle=True,
                expected_datasets={"finance", "shougang"},
                expected_grading_hashes=expected_grading_hashes,
            )
        except ValueError as exc:
            raise ValueError(f"prompt-identifiability bundle rejected: {exc}") from exc
        if verified_prompt["sha256"] != expected_prompt_sha or prompt.get("datasets") != ["finance", "shougang"]:
            raise ValueError("reference provenance prompt bundle lineage mismatch")
    else:
        raise ValueError("reference provenance requires a serialized prompt-identifiability bundle")
    return provenance


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint-dir", required=True,
                        help="Merged HF reference model directory loaded by RL")
    parser.add_argument("--export-report", required=True,
                        help="Passed/approved-waived SFT export_report.json")
    parser.add_argument("--effective-config", required=True,
                        help="Effective VeRL Hydra config saved for this run")
    parser.add_argument("--base-model", required=True,
                        help="Exact base HF model directory used by SFT")
    parser.add_argument("--environment-report", required=True,
                        help="JSON manifest of the training Python/CUDA environment")
    parser.add_argument(
        "--prompt-audit-report",
        help="Passed field-only prompt identifiability report or bundle",
    )
    parser.add_argument(
        "--prompt-audit-bundle", dest="prompt_audit_bundle",
        help="Alias for --prompt-audit-report when using the joint bundle",
    )
    parser.add_argument(
        "--grading-manifest", required=True,
        help="Verified finance+shougang grading-standard manifest",
    )
    parser.add_argument("--git-commit", required=True,
                        help="Training source commit (7-40 hex characters)")
    parser.add_argument("--global-step", required=True, type=int,
                        help="Selected fixed final training step")
    parser.add_argument("--output", required=True,
                        help="Destination reference provenance JSON (atomic write)")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        print(f"output already exists: {output} (pass --overwrite)", file=sys.stderr)
        return 2
    prompt_audit = args.prompt_audit_bundle or args.prompt_audit_report
    if not prompt_audit:
        print("prompt audit report or bundle is required", file=sys.stderr)
        return 2
    if args.prompt_audit_bundle and args.prompt_audit_report:
        print("provide only one prompt audit option", file=sys.stderr)
        return 2
    try:
        provenance = build_provenance(
            args.checkpoint_dir,
            args.export_report,
            effective_config=args.effective_config,
            base_model=args.base_model,
            environment_report=args.environment_report,
            prompt_audit_report=prompt_audit,
            git_commit=args.git_commit,
            global_step=args.global_step,
            grading_manifest=args.grading_manifest,
        )
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
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"record_checkpoint: status:error type={type(exc).__name__}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "checkpoint_sha256": provenance["checkpoint_sha256"],
                "files": provenance["files"],
                "global_step": provenance["global_step"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
