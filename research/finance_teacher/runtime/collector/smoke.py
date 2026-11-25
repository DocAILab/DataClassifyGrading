"""Run one TRAIN field through the current Bailian teacher, or replay a saved run."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from classification_protocol import identity, use_protocol

ROOT = Path(__file__).resolve().parent
R13 = ROOT.parent / "finance_teacher_coverage_trial_20261025_r13"
from release_inputs import manifest_sha
FREEZE = manifest_sha()
sys.path.insert(0, str(R13))
import r13_common


def exact(row):
    # A Windows junction can give the same frozen file two absolute names.
    path = Path(row["path"])
    if not path.is_absolute() or r13_common.binding(path) != {**row, "path": str(path.resolve(strict=True))}:
        raise ValueError(f"Frozen file identity differs: {path}")


# Historical modules import this function by name. Install it before importing them,
# in this isolated CLI process, so the collector and the independent replay agree.
r13_common.exact = exact
import r14_common as common
import r14_science as science
import r14_wire as wire
import scope_runtime
import table_observation
import source_evidence_protocol
import smoke_decoding
import train_table_catalog
import role_evidence
import review_abstention
import evidence_followup
from release_inputs import use_catalog
import table_recovery
import row_evidence
import field_roles
from abstention import apply_review, decision, verified_abstention


def verify_inputs():
    if common.verify_freeze() != FREEZE:
        raise ValueError("Release file manifest changed")
    return science.bundle()


from contextlib import contextmanager


@contextmanager
def protocols(admission, data):
    with (use_protocol(science.runtime(), admission.get("classification_protocol")),
          table_observation.use_format(science.runtime(), admission.get("table_observation_protocol")),
          scope_runtime.use_scope(science.native200, admission.get("scope_binding_protocol")),
          source_evidence_protocol.use(science, admission.get("source_evidence_protocol")),
          use_catalog(admission, source_evidence_protocol, train_table_catalog),
          role_evidence.use(science.runtime(), admission.get("role_evidence_protocol")),
          evidence_followup.use(science.runtime(), admission.get("followup_evidence_protocol"), data["study"]["registry"]),
          table_recovery.use(science.runtime(), admission.get("table_recovery_protocol")),
          row_evidence.use(science.runtime(), admission.get("row_evidence_protocol")),
          field_roles.use(science.runtime(), admission.get("field_roles_protocol"))):
        yield


def worker(target):
    admission = common.read(target / "admission.json")
    data = verify_inputs()
    with protocols(admission, data):
        return collect(target, admission, data)


def collect(target, admission, data):
    case = next(c for c in data["cases"] if c["source_index"] == admission["source_index"] and c["split"] == "train")
    config = common.read(ROOT / "config.json")
    thinking_budget = admission.get("smoke_thinking_budget")
    if thinking_budget is not None:
        config["thinking_budget"] = thinking_budget
    key = os.environ.pop("DASHSCOPE_API_KEY")
    camp = common.load("smoke_campaign", common.R9 / "campaign.py")
    runtime = science.runtime()
    collector = common.load("smoke_collector", ROOT.parent / "finance_teacher_program_coldstart_20261024_r4/collect_program.py")
    smoke_decoding.configure(collector, thinking_budget)
    collector.CommonActor = runtime.CommonActor
    sys.path.insert(0, str(ROOT / "reviews/resume04"))
    from r04_policy import budget_type
    budget = budget_type(camp.Budget)(admission["request_cap"], admission["cost_cap_CNY"], admission["deadline_unix"])
    engine, scoped, _, _ = runtime.native_context()
    common.put(target / "program-inputs.json", {"study": data["study"], "cases": [case]})
    common.put(target / "config.json", config)
    folder = target / "runs" / f"source-{case['source_index']:03d}"
    backend = collector.Backend(folder, config, budget, wire.capture_sse, key, common.verify_freeze)
    closed = False
    error = None
    try:
        rollout = science.native200.build(engine, scoped, lambda: backend.halted, data)
        trace = rollout(case["case"]["item"], data["study"], backend, copy.deepcopy(case["case"]["snapshot"]))
        common.put(folder / "trace.json", trace)
        closed = not budget.halted and trace["stop_reason"] != "campaign_budget_exhausted"
    except Exception as exc:
        error = type(exc).__name__
    code = 0 if closed else 1
    common.put(target / "completion.json", dict(actual_worker_exit=code, closed=int(closed),
               fixed_denominator=1, budget=budget.snapshot(), error_type=error,
               TestOnly=False, training_ready=False))
    return code


def verify(target):
    data = verify_inputs()
    admission = common.read(target / "admission.json")
    with protocols(admission, data):
        with smoke_decoding.verifier(science, admission.get("smoke_thinking_budget")):
            proof = science.verify(target, target / "verification.json")["summary"]
    print(json.dumps(proof, ensure_ascii=True), flush=True)
    return (proof["actual_OS_exit"] == 0 and proof["completed"] == 1 and
            proof["actual_model_calls"] > 0 and proof["wire_complete_all"] and
            proof["raw_JSON_objects"] == proof["actual_model_calls"] and
            proof["same_information_exact"] and proof["unknown_usage"] == 0)


def read_decision(target, verified, receipt="verification.json"):
    admission = common.read(target / "admission.json")
    folder = target / "runs" / f"source-{admission['source_index']:03d}"
    trace = common.read(folder / "trace.json") if (folder / "trace.json").is_file() else {}
    if not verified and (target / receipt).is_file():
        verified = verified_abstention(trace, common.read(target / receipt))
    registry = common.read(target / "program-inputs.json")["study"]["registry"] if verified else {}
    result = decision(trace, registry, verified)
    if result["prediction_source"] == "best_candidate" and (target / "review" / "result.json").exists():
        review = review_abstention.verify_review(target, target / "review")
        result = apply_review(result, review)
    return result


def save_decision(target, passed):
    result = read_decision(target, passed)
    # Persist the local answer before an optional paid call can fail or be interrupted.
    common.put(target / "decision.json", result)
    if result["prediction_source"] == "best_candidate":
        ledger = common.read(target / "completion.json")["budget"]
        remaining = min(180, ledger["deadline"] - time.time())
        if ledger["claims"] < ledger["request_cap"] and remaining >= 1:
            review_abstention.run(target, target / "review", ledger["cost_cap"] - ledger["spent"], remaining)
            result = read_decision(target, passed)
            staged = target / "decision.reviewed.partial"
            common.put(staged, result)
            os.replace(staged, target / "decision.json")
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("mode", choices=["run", "worker", "verify", "decision"])
    cli.add_argument("output", type=Path)
    cli.add_argument("--source-index", type=int)
    args = cli.parse_args()
    target = args.output.resolve()
    if args.mode == "verify":
        return 0 if verify(target) else 1
    if args.mode == "decision":
        passed = verify(target)
        result = read_decision(target, passed)
        print(json.dumps(result, ensure_ascii=True))
        return 0 if result["status"] == "classified" else 1
    if args.mode == "worker":
        with wire.isolation.lease(ROOT / "active-worker.lock"):
            return worker(target)
    if not common.read(ROOT / "config.json")["endpoint"]:
        cli.error("Set DASHSCOPE_BASE_URL to your Bailian workspace endpoint")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        cli.error("Supply DASHSCOPE_API_KEY in the environment")
    if not any(c["source_index"] == args.source_index and c["split"] == "train" for c in verify_inputs()["cases"]):
        cli.error("Choose one TRAIN source from the R14 manifest with --source-index")
    if target.is_relative_to(ROOT):
        cli.error("Choose a new output directory outside the historical campaign")
    # A cohort uses its predeclared protocol, including when the collector is
    # upgraded between two sources. Standalone experiments use the current version.
    protocols = dict(classification_protocol=identity(), scope_binding_protocol=scope_runtime.identity(),
                     table_observation_protocol=table_observation.identity(), source_evidence_protocol=source_evidence_protocol.identity(),
                     smoke_thinking_budget=512, train_table_catalog=train_table_catalog.identity(), role_evidence_protocol=role_evidence.identity(),
                     followup_evidence_protocol=evidence_followup.identity(), table_recovery_protocol=table_recovery.identity(),
                     row_evidence_protocol=row_evidence.identity(), field_roles_protocol=field_roles.identity())
    plan_file = target.parent / "plan.json"
    if plan_file.is_file() and "scope_binding_protocol" in (plan := common.read(plan_file)):
        if args.source_index not in plan["source_indices"]:
            cli.error("Source is outside the declared cohort")
        protocols = {key: plan.get(key) for key in protocols}
    target.mkdir(parents=True, exist_ok=False)
    deadline = time.time() + 1200
    common.put(target / "admission.json", dict(source_index=args.source_index, deadline_unix=deadline,
               request_cap=16, cost_cap_CNY=6.0, frozen_manifest_sha256=FREEZE,
               **protocols))
    cmd = [sys.executable, "-B", str(Path(__file__).resolve()), "worker", str(target)]
    exit_record = wire.isolation.run(cmd, target, deadline + 30)
    common.put(target / "OS-exit.json", exit_record)
    if not (target / "completion.json").is_file():
        raise RuntimeError(f"Worker exited before completion; see {target / 'stderr.log'}")
    passed = verify(target)
    result = save_decision(target, passed)
    print(json.dumps(result, ensure_ascii=True), flush=True)
    return 0 if passed and not result.get("review_accounting", {}).get("unknown_usage") else 1


if __name__ == "__main__":
    sys.exit(main())
