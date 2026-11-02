"""Thin bridge to real frozen binding, executor, publication and terminal guards."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

from . import _vendor
from .contracts import json_snapshot
from .failure_policy import COMPARISON
from .validation import SELECTION, GRADE
from agent.orchestration.agent_loop import LoopConfig, PolicyDecision
from agent.orchestration.category_catalog import CategoryCatalog
from agent.orchestration.runtime import Observation
from agent.orchestration.skill_loader import SkillLoader
from agent.orchestration.validator import resolve_reference
from pool_catalog import pool_catalog
from pool_metadata import PoolMetadata
from pool_runtime import PoolController, PoolExecutor, publish_state, SEMANTIC_KEYS, ALIGNMENT_KEYS
from pool_metadata import FIELDS
import pool_availability


class ExecutionBridge:
    def __init__(self, generate, registry, rows, config, *, include_comparison_explanation=True, skill_pool="P14"):
        self.catalog = pool_catalog(skill_pool)
        self.loader = SkillLoader(_vendor.VENDOR / "runtime" / "skills" / "full_pool", self.catalog, strict_catalog=False)
        self.loader.validate_all()
        self.executor = PoolExecutor(generate, self.loader, PoolMetadata(rows), CategoryCatalog(registry),
                                     include_comparison_explanation=include_comparison_explanation)
        self.guard = PoolController(policy=SimpleNamespace(missing_awareness=True), executor=self.executor,
                                    catalog=self.catalog, skill_loader=self.loader, config=LoopConfig(recover_ready_finalization=False),
                                    registry_entries=registry, approved_levels={e["grading_level"] for e in registry.values()}, state_publication=publish_state)
        self.availability = pool_availability
        self.max_identical_calls = config.max_identical_calls

    @staticmethod
    def _drop(state, keys):
        removed = sorted(keys & set(state))
        for key in removed:
            state.pop(key, None)
            state.get("_provenance", {}).pop(key, None)
        return removed

    def step(self, call, record, state, counts, lineage, comparison_failures=None):
        result = {"call": call.to_dict(), "status": "blocked", "binding": None, "invalidations": []}
        canonical = None
        comparison_fingerprint = None
        try:
            spec = self.catalog[call.tool]
            decision = self.guard._normalize_policy_decision(PolicyDecision(call, ""), record, state, spec)
            canonical = decision.call
            result["binding"] = decision.metadata["binding"]
            resolved = {k: resolve_reference(v, record, state)[1] for k, v in canonical.arguments.items()}
            identity = json.dumps({"call": {"tool": call.tool, "arguments": resolved}, "lineage": lineage}, sort_keys=True, ensure_ascii=False, allow_nan=False)
            if counts[identity] >= self.max_identical_calls:
                raise ValueError("identical successful call blocked for current evidence lineage")
            correction_context = None
            if call.tool == COMPARISON and comparison_failures is not None:
                inputs = self.executor.comparison_inputs(record, state, resolved)
                comparison_fingerprint, correction_context = comparison_failures.before_call(inputs)
            result["status"] = "error"
            if call.tool == COMPARISON:
                observation = self.executor.execute(canonical, record, state, self.loader.load(call.tool),
                                                    correction_context=correction_context)
            else:
                observation = self.executor.execute(canonical, record, state, self.loader.load(call.tool))
            # Normalize/availability update is transactional with publication.
            updated = deepcopy(state)
            observation = self.guard._normalize_observation(canonical, observation, updated)
            self.executor._validate_deterministic(observation, spec)
            result.update(asdict(observation))
            if observation.status == "success":
                changed = {key for key, value in observation.data.items() if state.get(key) != value}
                evidence = changed & (set(SEMANTIC_KEYS) | set(ALIGNMENT_KEYS) | set(FIELDS)
                                      | {"field_metadata", "acquired_metadata", "table_fields", "selected_context_fields",
                                         "context_fields", "context_field_descriptions", "context_field_types",
                                         "context_field_values", "table_context", "database_context"})
                invalidated = []
                if evidence or "standard_candidates" in changed:
                    invalidated.extend(self._drop(updated, SELECTION))
                elif "category" in changed or "aligned_standard_entry" in changed:
                    invalidated.extend(self._drop(updated, GRADE))
                if spec.terminal and not self.guard._valid_terminal(observation.data.get("final_result"), updated):
                    raise ValueError("invalid terminal result rejected by real PoolController")
                publish_state(record, updated, observation.data, call.tool)
                state.clear(); state.update(updated)
                # Global success dedup consumes the input lineage, before incrementing it.
                counts[identity] += 1
                if evidence: lineage["evidence"] += 1
                if "standard_candidates" in changed: lineage["candidates"] += 1
                if "category" in changed or "aligned_standard_entry" in changed: lineage["selection"] += 1
                result["invalidations"] = invalidated
            else:
                state.clear(); state.update(updated)
        except Exception as exc:
            if result["status"] == "success": result["status"] = "error"
            metadata = {**result.get("metadata", {}), **deepcopy(getattr(exc, "metadata", {}))}
            result.update(summary=str(exc), data={}, metadata=metadata)
            if comparison_fingerprint is not None and result["status"] == "error":
                comparison_failures.record_failure(comparison_fingerprint, exc, metadata)
                state["comparison_failure_ledger"] = comparison_failures.snapshot()
            if canonical is not None and result["status"] == "error":
                self.availability.observe(canonical, Observation(call.tool, "error", str(exc), {}), state)
        result["lineage"] = dict(lineage)
        return json_snapshot(result)
