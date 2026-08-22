"""Synthetic tests for the shared source-seeded candidate policy."""

from __future__ import annotations

import pytest

from agent.task import LeafRegistry
from agent.training.common import build_candidates


def _registry() -> LeafRegistry:
    return LeafRegistry.from_mapping(["A", "B", "C", "D", "E", "F"])


def test_same_source_id_yields_the_same_ordering() -> None:
    registry = _registry()
    assert build_candidates("C", registry, source_id="row-train") == build_candidates(
        "C", registry, source_id="row-train"
    )


def test_different_source_ids_can_yield_different_orderings() -> None:
    registry = _registry()
    bundles = {
        tuple(build_candidates("C", registry, source_id=sample_id))
        for sample_id in ("r1", "r2", "r3", "r4", "r5")
    }
    assert len(bundles) > 1


def test_ground_truth_always_present_and_exactly_five_unique() -> None:
    registry = _registry()
    for sample_id in ("row-train", "a", "b", "c", "d"):
        bundle = build_candidates("C", registry, source_id=sample_id)
        assert len(bundle) == 5
        assert len(set(bundle)) == 5
        assert "C" in bundle
        assert all(candidate in registry.ids for candidate in bundle)


def test_bundle_is_a_permutation_of_the_gt_first_slots() -> None:
    registry = _registry()
    base = ["C"] + [category_id for category_id in registry.ids if category_id != "C"][:4]
    for sample_id in ("row-train", "row-val", "row-test", "x", "y"):
        bundle = build_candidates("C", registry, source_id=sample_id)
        assert sorted(bundle) == sorted(base)


def test_ground_truth_position_is_not_fixed_at_one() -> None:
    registry = _registry()
    positions = {
        build_candidates("C", registry, source_id=sample_id).index("C")
        for sample_id in ("row-train", "row-val", "row-test", "r1", "r2")
    }
    assert any(position != 0 for position in positions)
    assert 0 in positions  # position 1 is legal, just not universal


def test_all_five_positions_appear_across_large_synthetic_id_set() -> None:
    registry = _registry()
    positions = {
        build_candidates("C", registry, source_id=f"sample-{index:04d}").index("C")
        for index in range(400)
    }
    assert positions == {0, 1, 2, 3, 4}


def test_source_id_is_a_required_keyword() -> None:
    registry = _registry()
    with pytest.raises(TypeError):
        build_candidates("C", registry)  # missing required source_id
