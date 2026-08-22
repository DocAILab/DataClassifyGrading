"""Target resolver interface and the deterministic classification resolver.

The resolver derives a canonical SampleTarget from one processed record while
leaving classification.level_1..level_4 untouched as provenance. It never
auto-fixes labels and never uses semantic matching.

Resolution outcomes are explicit (ResolutionStatus / ResolutionResult):
- INVALID_RECORD: no classification, non-object classification, or empty leaf;
- UNLABELED: label_status == "unlabeled" (no target, even when a leaf exists);
- PLACEHOLDER: the leaf is one of the configured placeholder labels;
- CODE_UNRESOLVED: code strategy and the corpus carries no code for the leaf
  (no silent fallback to another ID scheme);
- RESOLVED: a SampleTarget was generated. Registry membership is NOT checked
  here — the canonical dataset layer verifies category_id against the
  LeafRegistry and splits RESOLVED into registry-covered / missing_leaf /
  path_mismatch.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from agent.task.contracts import SampleTarget
from agent.task.dataset_config import DatasetConfig
from agent.task.identity import qualified_category_id


class ResolutionStatus(str, Enum):
    """Explicit resolution outcome for one record."""

    RESOLVED = "resolved"
    UNLABELED = "unlabeled"
    PLACEHOLDER = "placeholder"
    MISSING_LEAF = "missing_leaf"
    PATH_MISMATCH = "path_mismatch"
    CODE_UNRESOLVED = "code_unresolved"
    INVALID_RECORD = "invalid_record"


@dataclass(frozen=True)
class ResolutionResult:
    """One record's resolution outcome.

    status RESOLVED with a target: the target still needs registry-membership
    verification (done by the canonical dataset layer, which may downgrade it
    to MISSING_LEAF / PATH_MISMATCH).

    ``leaf`` / ``category_id`` are minimal audit fields populated whenever
    the information is available (e.g. CODE_UNRESOLVED carries the leaf but
    no category_id). Consumers must never reach into ``target`` for
    downgraded / unresolved statuses — use these audit fields instead.
    """

    status: ResolutionStatus
    target: SampleTarget | None = None
    reason: str = ""
    leaf: str = ""
    category_id: str = ""


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
        """Number of records skipped structurally (no classification, empty
        leaf, or placeholder label)."""
        ...

    @property
    def unresolved(self) -> Mapping[str, int]:
        """leaf name -> count for records that resolved to no target because
        the code strategy found no code for the leaf (incomplete corpus)."""
        ...


@dataclass
class ClassificationTargetResolver:
    """Deterministic resolver for normalized classification records.

    id_strategy "code": category_id = code_leaf_map[leaf_name]; a leaf
    without a code is unresolved (no target, reported, never silently
    remapped to another ID scheme).
    id_strategy "path": category_id = qualified_category_id(dataset,
    identity_fields values in canonical order); human-readable and
    path-qualified, so the same leaf name under different parents never
    collides. ``identity_fields`` may exclude provenance-only levels when a
    local configuration defines a smaller canonical identity.
    """

    config: DatasetConfig
    code_leaf_map: Mapping[str, str] = field(default_factory=dict)
    resolved: int = 0
    skipped: int = 0
    unresolved_leaves: Counter[str] = field(default_factory=Counter)

    def resolve(self, record: Mapping[str, Any]) -> SampleTarget | None:
        """Return the canonical target for one processed record, or None when
        the record must not produce a training target."""
        return self.resolve_detailed(record).target

    def resolve_detailed(self, record: Any) -> ResolutionResult:
        """Resolve with an explicit outcome status (see ResolutionStatus).

        Non-Mapping records (list items, strings, None) resolve to
        INVALID_RECORD instead of raising AttributeError.
        """
        if not isinstance(record, Mapping):
            self.skipped += 1
            return ResolutionResult(
                ResolutionStatus.INVALID_RECORD,
                reason="record is not an object",
            )
        classification = record.get("classification")
        if not isinstance(classification, Mapping):
            self.skipped += 1
            return ResolutionResult(
                ResolutionStatus.INVALID_RECORD,
                reason="classification missing or not an object",
            )

        levels = [
            str(classification.get(field, "") or "").strip()
            for field in self.config.path_fields
        ]
        leaf_index = self.config.path_fields.index(self.config.leaf_level)
        leaf = levels[leaf_index]

        if str(record.get("label_status", "") or "").strip().lower() == "unlabeled":
            self.skipped += 1
            return ResolutionResult(
                ResolutionStatus.UNLABELED,
                reason="label_status is unlabeled",
                leaf=leaf,
            )
        if not leaf:
            self.skipped += 1
            return ResolutionResult(
                ResolutionStatus.INVALID_RECORD,
                reason="leaf is empty",
            )
        if leaf in self.config.placeholder_labels:
            self.skipped += 1
            return ResolutionResult(
                ResolutionStatus.PLACEHOLDER,
                reason=f"placeholder label {leaf!r}",
                leaf=leaf,
            )

        if self.config.id_strategy == "code":
            category_id = self.code_leaf_map.get(leaf)
            if category_id is None:
                self.unresolved_leaves[leaf] += 1
                return ResolutionResult(
                    ResolutionStatus.CODE_UNRESOLVED,
                    reason=f"no code for leaf {leaf!r} in the corpus",
                    leaf=leaf,
                )
        else:
            identity_fields = self.config.identity_fields or self.config.path_fields
            identity_indices = [
                self.config.path_fields.index(field) for field in identity_fields
            ]
            category_id = qualified_category_id(
                self.config.dataset, [levels[index] for index in identity_indices]
            )

        category_path = tuple(part for part in levels if part)
        self.resolved += 1
        return ResolutionResult(
            ResolutionStatus.RESOLVED,
            target=SampleTarget(
                leaf_level=self.config.leaf_level,
                leaf_name=leaf,
                category_id=category_id,
                category_path=category_path,
            ),
            leaf=leaf,
            category_id=category_id,
        )

    @property
    def unresolved(self) -> Mapping[str, int]:
        return dict(self.unresolved_leaves)


def resolve_all(
    records: Sequence[Mapping[str, Any]],
    resolver: TargetResolver,
) -> tuple[list[SampleTarget], list[dict[str, Any]]]:
    """Resolve a batch; returns (targets, skipped_records).

    Skipped records include structural skips and unresolved code lookups;
    inspection of ``resolver.unresolved`` distinguishes the two.
    """
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
