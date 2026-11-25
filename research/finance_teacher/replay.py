"""Replay the 25 original provider trajectories with the native teacher, without an API key."""
import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "runtime/collector"))
import smoke
from release_inputs import verify_files
from score_smoke import prediction_totals, score_prediction


def read(path):
    return json.loads(path.read_bytes())


class RecordedBackend:
    def __init__(self, rows, actor, campaign, config):
        self.rows = rows
        self.actor = actor
        self.campaign = campaign
        self.config = config
        self.position = 0
        self.last_call_evidence = {}
        self.last_generation_metadata = {}

    def __call__(self, native_messages):
        expected = self.rows[self.position]
        projected = self.actor.project(native_messages)
        body = self.campaign.build_body(projected["messages"], self.config)
        body["thinking_budget"] = self.config["thinking_budget"]
        if (self.campaign.message_sha(native_messages) != expected["native_input_sha256"]
                or projected["input_sha256"] != expected["input_sha256"]
                or body != expected["body"]
                or self.campaign.sha(self.campaign.canon(body)) != expected["body_sha256"]):
            raise ValueError(f"Different model input at call {self.position + 1}")
        raw = expected["provider_raw_content"]
        if expected["backend_exception"] or self.campaign.sha(raw.encode()) != expected["provider_target_sha256"]:
            raise ValueError(f"Provider response mismatch at call {self.position + 1}")
        self.position += 1
        self.last_call_evidence = expected["evidence"]
        self.last_generation_metadata = expected["metadata"]
        return raw


def replay(row, data, case, rt, campaign):
    path = ROOT / row["trace"]
    raw_trace = gzip.decompress(path.read_bytes())
    if hashlib.sha256(raw_trace).hexdigest() != row["trace_sha256"]:
        raise ValueError(f"Trace digest differs: {path}")
    archive = json.loads(raw_trace)
    decisions = read(ROOT / row["decision_file"])
    admission = read(ROOT / row["protocol"])
    calls = [json.loads(line) for line in gzip.decompress((ROOT / row["calls"]).read_bytes()).splitlines()]
    config = smoke.common.read(smoke.ROOT / "config.json")
    config["thinking_budget"] = admission["smoke_thinking_budget"]
    with smoke.protocols(admission, data):
        backend = RecordedBackend(calls, rt.CommonActor(), campaign, config)
        engine, scoped, _, _ = rt.native_context()
        rollout = smoke.science.native200.build(engine, scoped, lambda: False, data)
        reproduced = rollout(case["case"]["item"], data["study"], backend, copy.deepcopy(case["case"]["snapshot"]))
    if archive != reproduced or backend.position != len(calls) or archive["source_id"] != row["source_id"]:
        raise ValueError(f"Native execution differs: {path}")
    native = smoke.decision(reproduced, data["study"]["registry"], True)
    review_path = path.parent / "review.json"
    if review_path.is_file():
        review = read(review_path)
        gap = reproduced["gap_ledger"][-1]
        call = next(c for c in reversed(calls) if c["stage"] == gap["skill"]
                    and c["provider_target_sha256"] == gap["provider_response_sha256"])
        registry = data["study"]["registry"]
        messages, mapping, facts = smoke.review_abstention.review_input(reproduced, gap, call, registry)
        _, _, body = smoke.review_abstention.request_body(messages)
        if body != review["body"]:
            raise ValueError(f"Independent review input differs: {path}")
        proposed = smoke.review_abstention.suggestion(review["content"], mapping, registry, facts)
        if proposed != review["receipt"]["suggestion"]:
            raise ValueError(f"Independent review result differs: {path}")
        native = smoke.apply_review(native, review["receipt"])
    if (decisions != native or row["prediction"] != native["prediction"]
            or row["decision_status"] != native["status"] or row["prediction_source"] != native["prediction_source"]):
        raise ValueError(f"Final classification differs: {path}")
    matches, status = score_prediction(native, row["reference"])
    if status != row["status"] or any(row[k] != v for k, v in matches.items()):
        raise ValueError(f"Field score differs: {path}")
    eligible = native["prediction_source"] == "evidence" and matches["joint_match"] is True
    if row["eligible_for_sft_review"] != eligible:
        raise ValueError(f"SFT review eligibility differs: {path}")
    return len(calls)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("sources", nargs="*", type=int, help="Optional TRAIN source indices to replay")
    args = cli.parse_args()
    verify_files()
    data = smoke.science.bundle()
    by_index = {c["source_index"]: c for c in data["cases"] if c["split"] == "train"}
    if len(by_index) != 25 or set(args.sources) - by_index.keys():
        cli.error("Choose indices from the 25 published TRAIN fields")
    rt = smoke.science.runtime()
    campaign = smoke.common.load("release_replay_campaign", smoke.common.R9 / "campaign.py")
    scores = read(ROOT / "scores.json")
    rows = scores["rows"]
    refs = read(ROOT / "inputs/references.json")["labels"]
    if any(row["reference"] != refs[row["source_id"]] for row in rows):
        raise ValueError("Published references disagree")
    if scores["combined25"] != prediction_totals(rows):
        raise ValueError("Aggregate score differs")
    if {r["source_index"] for r in rows} != by_index.keys() or len(rows) != len(by_index):
        raise ValueError("Published score and input cohorts differ")
    selected = [r for r in rows if not args.sources or r["source_index"] in args.sources]
    total = 0
    for row in selected:
        total += replay(row, data, by_index[row["source_index"]], rt, campaign)
    correct = sum(r["joint_match"] for r in selected)
    print(f"{len(selected)} traces and {total} model calls matched. "
          f"{correct}/{len(selected)} final classifications correct. 0 network requests.")


if __name__ == "__main__":
    main()
