"""Score a completed, preselected teacher run offline using local TRAIN references."""
import argparse
from collections import Counter
from pathlib import Path

from smoke import common, read_decision, verify


def score_prediction(result, reference):
    matches = {"category_match": None, "grade_match": None, "joint_match": None}
    if reference is None:
        return matches, "missing_reference"
    if result["status"] != "classified":
        return matches, "no_valid_prediction"
    prediction = result["prediction"]
    matches["category_match"] = prediction["category"] == reference["category"]
    matches["grade_match"] = prediction["grading_level"] == reference["grading_level"]
    matches["joint_match"] = matches["category_match"] and matches["grade_match"]
    return matches, "reference_match" if matches["joint_match"] else "reference_mismatch"


def prediction_totals(rows):
    return {"sources": len(rows), "classified": sum(r["decision_status"] == "classified" for r in rows),
            "joint_matches": sum(r["joint_match"] is True for r in rows),
            "best_candidate": sum(r["prediction_source"] == "best_candidate" for r in rows),
            "sft_candidates": sum(r["eligible_for_sft_review"] for r in rows)}


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
        matches, status = score_prediction(result, ref)
        if not summary["wire_complete_all"] or summary["unknown_usage"]:
            status = "incomplete_transport"
        elif row["stop_reason"] == "campaign_budget_exhausted":
            status = "budget_exhausted"
        elif result["status"] == "incomplete" and summary["raw_JSON_objects"] != summary["actual_model_calls"]:
            status = "invalid_json"
        elif result["status"] == "incomplete" and not passed:
            status = "execution_failure"
        trace = target / "runs" / f"source-{row['source_index']:03d}" / "trace.json"
        rows.append({**{k: row[k] for k in ("source_index", "source_id", "stop_reason")},
                     "native_legal": row["legal"], "prediction": result["prediction"],
                     "decision_status": result["status"], "prediction_source": result["prediction_source"],
                     "review_status": result.get("review_status"),
                     "review_accounting": result.get("review_accounting", {}),
                     "missing_evidence": result["missing_evidence"],
                     "reference": ref, **matches, "status": status, "eligible_for_sft_review": passed and status == "reference_match" and result["prediction_source"] == "evidence",
                     "native_replay_exact": row["native_replay_exact"], "wire_complete": summary["wire_complete_all"],
                     "raw_json_objects": summary["raw_JSON_objects"], "unknown_usage": summary["unknown_usage"],
                     "http_calls": summary["actual_model_calls"], "estimated_CNY": summary["conservative_cost_CNY"],
                     "trace": str(trace.relative_to(directory)) if trace.exists() else None})
    return {"model": plan["model"], **prediction_totals(rows),
            "native_replays": sum(r["native_replay_exact"] for r in rows),
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
    print(f"{result['classified']}/{result['sources']} classified; {result['joint_matches']} correct category and grade.")
