"""Target resolver interface and the deterministic classification resolver.

The resolver derives a canonical SampleTarget from one processed record while
leaving classification.level_1..level_4 untouched as provenance. It never
auto-fixes labels and never uses semantic matching; records whose leaf is a
placeholder label or missing resolve to None (to be filtered by the pipeline).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from agent.task.contracts import SampleTarget
from agent.task.dataset_config import DatasetConfig
from agent.task.identity import stable_category_id


class TargetResolver(Protocol):
    """Interface every target resolver must satisfy."""

    def resolve(self, record: Mapping[str, Any]) -> SampleTarget | None:
        """Return the canonical target for one processed record, or None when
        the record must not produce a training target."""
        ...

    @property
    def resolved(self) -> int:
        """Number of records that produced a target."""
        ...

    @property
    def skipped(self) -> int:
        """Number of records that produced no target."""
        ...

    @property
    def code_fallbacks(self) -> int:
        """Number of records whose category_id fell back to the path hash
        because the code map had no entry for the leaf (incomplete corpus)."""
        ...


@dataclass
class ClassificationTargetResolver:
    """Deterministic resolver for the normalized TransClass record format.

    id_strategy "code": category_id = code_leaf_map[leaf_name]; when the leaf
    has no code (corpus incomplete), fall back to the path hash and count it.
    id_strategy "path_hash": category_id = stable_category_id(dataset, all
    path fields). Same leaf under different parents therefore never collides.
    """

    config: DatasetConfig
    code_leaf_map: Mapping[str, str] = field(default_factory=dict)
    resolved: int = 0
    skipped: int = 0
    code_fallbacks: int = 0

    def resolve(self, record: Mapping[str, Any]) -> SampleTarget | None:
        classification = record.get("classification")
        if not isinstance(classification, Mapping):
            self.skipped += 1
            return None

        levels = [
            str(classification.get(field, "") or "").strip()
            for field in self.config.path_fields
        ]
        leaf_index = self.config.path_fields.index(self.config.leaf_level)
        leaf = levels[leaf_index]
        if not leaf:
            self.skipped += 1
            return None
        if leaf in self.config.placeholder_labels:
            self.skipped += 1
            return None

        if self.config.id_strategy == "code":
            category_id = self.code_leaf_map.get(leaf)
            if category_id is None:
                category_id = stable_category_id(self.config.dataset, levels)
                self.code_fallbacks += 1
        else:
            category_id = stable_category_id(self.config.dataset, levels)

        category_path = tuple(part for part in levels if part)
        self.resolved += 1
        return SampleTarget(
            leaf_level=self.config.leaf_level,
            leaf_name=leaf,
            category_id=category_id,
            category_path=category_path,
        )


def resolve_all(
    records: Sequence[Mapping[str, Any]],
    resolver: TargetResolver,
) -> tuple[list[SampleTarget], list[dict[str, Any]]]:
    """Resolve a batch; returns (targets, skipped_records)."""
    targets: list[SampleTarget] = []
    skipped: list[dict[str, Any]] = []
    for record in records:
        target = resolver.resolve(record)
        if target is None:
            skipped.append(dict(record))
        else:
            targets.append(target)
    return targets, skipped


__all__ = [
    "TargetResolver",
    "ClassificationTargetResolver",
    "resolve_all",
]
