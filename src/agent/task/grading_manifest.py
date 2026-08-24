"""Verified per-dataset grading standards for finance+shougang joint runs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from agent.hashing import sha256_file
from .contracts import GradingConfig

_DATASETS = ("finance", "shougang")
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class DatasetGradingManifest:
    configs: Mapping[str, GradingConfig]
    hashes: Mapping[str, str]
    paths: Mapping[str, Path]
    source_path: Path

    def __post_init__(self) -> None:
        if set(self.configs) != set(_DATASETS):
            raise ValueError(
                "grading manifest configs must be exactly finance and shougang"
            )
        if set(self.hashes) != set(_DATASETS):
            raise ValueError(
                "grading manifest hashes must be exactly finance and shougang"
            )
        if set(self.paths) != set(_DATASETS):
            raise ValueError(
                "grading manifest paths must be exactly finance and shougang"
            )
        for dataset in _DATASETS:
            config = self.configs[dataset]
            if not isinstance(config, GradingConfig):
                raise ValueError(f"grading manifest {dataset} config is invalid")
            if config.gt_field != "data_level":
                raise ValueError(
                    f"formal grading standard for {dataset} must use gt_field data_level"
                )
            if not config.descriptions:
                raise ValueError(
                    f"formal grading standard for {dataset} requires descriptions"
                )
            digest = self.hashes[dataset]
            if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest):
                raise ValueError(f"grading manifest {dataset} sha256 is invalid")
            if not isinstance(self.paths[dataset], Path):
                raise ValueError(f"grading manifest {dataset} path is invalid")

    def config_for(self, dataset: str) -> GradingConfig:
        try:
            return self.configs[dataset]
        except KeyError as exc:
            raise KeyError(f"no grading standard for dataset {dataset!r}") from exc

    def sha256_for(self, dataset: str) -> str:
        try:
            return self.hashes[dataset]
        except KeyError as exc:
            raise KeyError(f"no grading standard hash for dataset {dataset!r}") from exc

    @classmethod
    def from_path(cls, path: str | Path) -> "DatasetGradingManifest":
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"grading manifest not found: {source}")

        def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"grading manifest contains duplicate key {key!r}")
                result[key] = value
            return result

        value = json.loads(
            source.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates
        )
        if not isinstance(value, Mapping) or set(value) != {"datasets"}:
            raise ValueError("grading manifest must contain only a datasets object")
        datasets = value.get("datasets")
        if not isinstance(datasets, Mapping) or set(datasets) != set(_DATASETS):
            raise ValueError("grading manifest datasets must be exactly finance and shougang")
        configs: dict[str, GradingConfig] = {}
        hashes: dict[str, str] = {}
        paths: dict[str, Path] = {}
        for dataset in _DATASETS:
            entry = datasets[dataset]
            if not isinstance(entry, Mapping) or set(entry) != {"path", "sha256"}:
                raise ValueError(
                    f"grading manifest {dataset} entry must contain path and sha256"
                )
            raw_path = entry["path"]
            digest = entry["sha256"]
            if not isinstance(raw_path, str) or not raw_path.strip():
                raise ValueError(f"grading manifest {dataset} path must be non-empty")
            if not isinstance(digest, str) or not _SHA_RE.fullmatch(digest):
                raise ValueError(f"grading manifest {dataset} sha256 is invalid")
            asset = Path(raw_path)
            if not asset.is_absolute():
                asset = source.parent / asset
            if not asset.is_file():
                raise FileNotFoundError(f"grading standard not found for {dataset}: {asset}")
            actual = sha256_file(asset)
            if actual != digest:
                raise ValueError(
                    f"grading standard sha256 mismatch for {dataset}: "
                    f"expected {digest}, got {actual}"
                )
            config = GradingConfig.from_path(asset)
            if config.gt_field != "data_level":
                raise ValueError(
                    f"formal grading standard for {dataset} must use gt_field data_level"
                )
            if not config.descriptions:
                raise ValueError(
                    f"formal grading standard for {dataset} requires descriptions"
                )
            configs[dataset] = config
            hashes[dataset] = actual
            paths[dataset] = asset
        return cls(configs=configs, hashes=hashes, paths=paths, source_path=source)


__all__ = ["DatasetGradingManifest"]
