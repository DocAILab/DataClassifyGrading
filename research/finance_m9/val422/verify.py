"""Rescore both Val422 runs from their predictions and separate reference labels."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_bytes())


def verify_scores(name, refs, expected):
    seen = set()
    totals = [0, 0, 0, 0]
    with (ROOT / name).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            sid = row["source_id"]
            if sid not in refs or sid in seen:
                raise ValueError(f"{name}: unexpected or repeated source {sid}")
            seen.add(sid)
            ref = refs[sid]
            if (ref["category"], ref["grading_level"]) != (row["reference_category"], row["reference_grading_level"]):
                raise ValueError(f"{name}: reference differs for {sid}")
            if row["validated_terminal"] not in {"0", "1"}:
                raise ValueError(f"{name}: invalid terminal flag for {sid}")
            valid = row["validated_terminal"] == "1"
            category = valid and row["predicted_category"] == ref["category"]
            grade = valid and row["predicted_grading_level"] == ref["grading_level"]
            counts = [valid, category, grade, category and grade]
            flags = [int(row[key]) for key in ("category_match", "grading_match", "joint_match")]
            if flags != counts[1:]:
                raise ValueError(f"{name}: stored score differs for {sid}")
            for i, value in enumerate(counts):
                totals[i] += value
    if seen != refs.keys() or totals != expected:
        raise ValueError(f"{name}: unexpected cohort or totals {totals}")
    print(f"{name}: {len(seen)} sources, {totals[0]} valid, {totals[1]} category, "
          f"{totals[2]} grade, {totals[3]} joint matches")


def main():
    refs = read("val422-references.json")["references"]
    inputs = read("val422-inputs.json")["rows"]
    study = read("nontraining-79/study.json")
    plans = read("m9-first-plans.json")
    for ids in ([row["source_id"] for row in inputs], study["cohorts"]["full422"],
                [row["id"] for row in study["rows"]], [sid for plan in plans for sid in plan["source_ids"]]):
        if len(ids) != 422 or set(ids) != refs.keys():
            raise ValueError("The inputs, plans and references must cover the same 422 sources")
    if any(p["count"] != len(p["source_ids"]) for p in plans):
        raise ValueError("Incorrect first-plan source count")
    verify_scores("m9-185-per-field.csv", refs, [410, 185, 230, 185])
    verify_scores("nontraining-79/per-field.csv", refs, [191, 79, 111, 79])


if __name__ == "__main__":
    main()
