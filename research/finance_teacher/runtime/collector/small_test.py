"""Run a fixed, serial set of TRAIN sources with the corrected teacher protocol."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from classification_protocol import identity
from smoke import ROOT, common, scope_runtime, source_evidence_protocol, table_observation, verify_inputs
import train_table_catalog
import role_evidence
import evidence_followup
import table_recovery
import row_evidence
import field_roles


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("sources", nargs="+", type=int)
    cli.add_argument("--thinking-budget", type=int, default=512, help="Use a reduced thinking budget for this cohort")
    cli.add_argument("--train-table-catalog", action="store_true", default=True, help="The published TRAIN projection includes this catalog")
    cli.add_argument("--evidence-updates", nargs="+", choices=evidence_followup.CHANGES, default=evidence_followup.CHANGES,
                     help="Apply individual evidence changes in a controlled comparison (default: all)")
    args = cli.parse_args()
    directory = args.output.resolve()
    if not common.read(ROOT / "config.json")["endpoint"]:
        cli.error("Set DASHSCOPE_BASE_URL to your Bailian workspace endpoint")
    if not os.environ.get("DASHSCOPE_API_KEY"):
        cli.error("Supply DASHSCOPE_API_KEY in the environment")
    if args.thinking_budget is not None and not 1 <= args.thinking_budget <= 2048:
        cli.error("Use a positive thinking budget up to the original 2048 tokens")
    available = {s["source_index"] for s in verify_inputs()["cases"] if s["split"] == "train"}
    if len(set(args.sources)) != len(args.sources) or set(args.sources) - available:
        cli.error("Choose distinct TRAIN sources")
    if directory.is_relative_to(ROOT):
        cli.error("Choose a new directory outside the historical campaign")
    directory.mkdir(parents=True, exist_ok=False)
    common.put(directory / "plan.json", {"model": common.read(ROOT / "config.json")["model"],
               "source_indices": args.sources, "started_unix": time.time(),
               "classification_protocol": identity(), "scope_binding_protocol": scope_runtime.identity(),
               "table_observation_protocol": table_observation.identity(), "source_evidence_protocol": source_evidence_protocol.identity(),
               "smoke_thinking_budget": args.thinking_budget,
               "role_evidence_protocol": role_evidence.identity(),
               "row_evidence_protocol": row_evidence.identity(),
               "field_roles_protocol": field_roles.identity(),
               "followup_evidence_protocol": evidence_followup.identity(args.evidence_updates),
               **({"table_recovery_protocol": table_recovery.identity()} if "context" in args.evidence_updates else {}),
               "train_table_catalog": train_table_catalog.identity() if args.train_table_catalog else None})
    all_passed = True
    for index in args.sources:
        target = directory / f"source-{index:03d}"
        result = subprocess.run([sys.executable, "-B", str(ROOT / "smoke.py"), "run",
                                 str(target), "--source-index", str(index)])
        if result.returncode:
            all_passed = False
            receipt = target / "verification.json"
            if not receipt.exists():
                return result.returncode
            summary = common.read(receipt)["summary"]
            review = target / "review"
            if review.exists() and (not (review / "result.json").exists()
                                    or common.read(review / "result.json")["unknown_usage"]):
                return result.returncode
            if summary["unknown_usage"] or not summary["wire_complete_all"]:
                return result.returncode
            # A recorded model-format rejection or a per-source budget end is a
            # sample outcome. Keep it in the denominator and run the next source.
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
