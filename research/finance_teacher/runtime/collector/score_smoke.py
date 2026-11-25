"""Score a completed, preselected teacher run offline using local TRAIN references."""
import argparse
from collections import Counter
from pathlib import Path

from smoke import common, read_decision, verify


def score_prediction(row, reference):
    prediction = row["prediction"] or {}
    check = {"category_match": None, "grade_match": None, "joint_match": None}
    if reference is None:
        return check, "missing_reference"
    if not row["legal"] or row["stop_reason"] != "validated_terminal":
        return check, ("insufficient_evidence" if row["stop_reason"] == "unresolved_no_new_business_evidence"
                       else "no_valid_prediction")
    check["category_match"] = prediction.get("category") == reference["category"]
    check["grade_match"] = prediction.get("grading_level") == reference["grading_level"]
    check["joint_match"] = check["category_match"] and check["grade_match"]
    return check, "reference_match" if check["joint_match"] else "reference_mismatch"


def summarize(directory, reference_file):
    plan = common.read(directory / "plan.json")
    indices = plan["source_indices"]
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Specify a nonempty set of distinct source indices")
    verified = []
    for index in indices:
        target = directory / f"source-{index:03d}"
        passed = verify(target)
        proof = common.read(target / "verification.json")
        row, = proof["rows"]
        admission = common.read(target / "admission.json")
        if plan.get("row_evidence_protocol") != admission.get("row_evidence_protocol"):
            raise ValueError(f"Source {index} used a different row-evidence protocol")
        if plan.get("field_roles_protocol") != admission.get("field_roles_protocol"):
            raise ValueError(f"Source {index} used a different field-qualifier protocol")
        if any(k in plan and plan[k] != admission.get(k)
               for k in ("scope_binding_protocol", "classification_protocol", "table_observation_protocol",
                         "source_evidence_protocol", "smoke_thinking_budget", "train_table_catalog", "role_evidence_protocol",
                         "followup_evidence_protocol", "table_recovery_protocol")):
            raise ValueError(f"Source {index} used a different experiment protocol")
        if (row["source_index"] != index or row["split"] != "train"
                or common.read(target / "config.json")["model"] != plan["model"]):
            raise ValueError(f"Unexpected source, split, or model in {target}")
        verified.append((row, proof["summary"], target, passed, read_decision(target, passed)))

    # Only the offline scorer opens the label file; the worker accepts no reference argument.
    refs = common.read(reference_file)["labels"]
    rows = []
    for row, summary, target, passed, result in verified:
        ref = refs.get(row["source_id"])
        matches, status = score_prediction(row, ref)
        if not summary["wire_complete_all"] or summary["unknown_usage"]:
            status = "incomplete_transport"
        elif row["stop_reason"] == "campaign_budget_exhausted":
            status = "budget_exhausted"
        elif summary["raw_JSON_objects"] != summary["actual_model_calls"]:
            status = "invalid_json"
        elif not passed:
            status = "execution_failure"
        trace = target / "runs" / f"source-{row['source_index']:03d}" / "trace.json"
        suggestion = result["suggestion"]
        suggestion_match = (all(suggestion[key] == ref[key] for key in ("category", "grading_level"))
                            if suggestion and ref else None)
        rows.append({**{k: row[k] for k in ("source_index", "source_id", "legal", "stop_reason", "prediction")},
                     "decision_status": result["status"],
                     "review_status": result.get("review_status"),
                     "suggestion": suggestion, "suggestion_joint_match": suggestion_match,
                     "review_accounting": result.get("review_accounting", {}),
                     "missing_evidence": result["missing_evidence"],
                     "reference": ref, **matches, "status": status, "eligible_for_sft_review": status == "reference_match",
                     "native_replay_exact": row["native_replay_exact"], "wire_complete": summary["wire_complete_all"],
                     "raw_json_objects": summary["raw_JSON_objects"], "unknown_usage": summary["unknown_usage"],
                     "http_calls": summary["actual_model_calls"], "estimated_CNY": summary["conservative_cost_CNY"],
                     "trace": str(trace.relative_to(directory)) if trace.exists() else None})
    return {"model": plan["model"], "sources": len(rows), "native_replays": sum(r["native_replay_exact"] for r in rows),
            "legal": sum(r["legal"] for r in rows), "joint_matches": sum(r["joint_match"] is True for r in rows),
            "suggestions": sum(r["suggestion"] is not None for r in rows),
            "abstentions": sum(r["decision_status"] == "abstained" for r in rows),
            "suggestion_joint_matches": sum(r["suggestion_joint_match"] is True for r in rows),
            "review_http_calls": sum(r["review_accounting"].get("http_calls", 0) for r in rows),
            "review_estimated_CNY": round(sum(r["review_accounting"].get("estimated_CNY", 0) for r in rows), 6),
            "http_calls": sum(r["http_calls"] for r in rows),
            "estimated_CNY": round(sum(r["estimated_CNY"] for r in rows), 6),
            "counts": dict(Counter(r["status"] for r in rows)), "rows": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--references", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.directory.resolve(), args.references)
    common.put(args.directory / "scores.json", result)
    print(f"{result['sources']} sources, {result['legal']} classified, {result['joint_matches']} joint matches; "
          f"{result['suggestions']} review suggestions, {result['suggestion_joint_matches']} suggestion joint matches")
