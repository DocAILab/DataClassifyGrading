"""Compare the final candidates once using the observed facts of a verified abstention."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

from abstention import review_status


class ReviewUnavailable(ValueError):
    """The verified trace has no evidence-bearing classifier call to compare."""


def prepare(target):
    from smoke import common
    row, = common.read(target / "verification.json")["rows"]
    folder = target / "runs" / f"source-{row['source_index']:03d}"
    trace = common.read(folder / "trace.json")
    if trace["stop_reason"] != "unresolved_no_new_business_evidence" or not trace.get("gap_ledger"):
        raise ReviewUnavailable("A closed evidence abstention is required")
    gap = trace["gap_ledger"][-1]
    for path in sorted(folder.glob("call-*/request.json"), reverse=True):
        request = common.read(path)
        if request["stage"] != gap.get("skill") or not path.with_name("outcome.json").is_file():
            continue
        outcome = common.read(path.with_name("outcome.json"))
        if outcome["provider_target_sha256"] and outcome["provider_target_sha256"] == gap.get("provider_response_sha256"):
            break
    else:
        raise ReviewUnavailable("Cannot bind the gap to its original classifier call")
    registry = common.read(target / "program-inputs.json")["study"]["registry"]
    messages, mapping, facts = review_input(trace, gap, request, registry)
    return messages, mapping, registry, facts, path


def review_input(trace, gap, request, registry):
    payload = json.loads(request["body"]["messages"][-1]["content"])
    facts = {}
    for key, entry in payload.get("evidence_boundary", {}).get("facts", {}).items():
        value = entry.get("value")
        if (entry.get("role") == "raw_fact" and entry.get("decisive_eligible") is True
                and isinstance(value, (str, int, float)) and str(value).strip()):
            facts[f"F{len(facts) + 1:02d}"] = {"source_id": key, "value": value}
    by_label = {" > ".join(v["path"]): k for k, v in registry.items()}
    options = sorted(payload.get("candidate_details", []), key=lambda c: c["option_id"])
    if any(c.get("label") not in by_label for c in options):
        raise ReviewUnavailable("Unknown category in the source classifier options")
    mapping = {c["option_id"]: by_label[c["label"]] for c in options}
    if not options or len(mapping) != len(options) or not facts:
        raise ReviewUnavailable("Missing review options or source facts")
    messages = [
        {"role": "system", "content":
         "你是数据库字段分类的复核员。正式分类已因证据不足而弃权。现在从给定候选中选一个最可能的类别供人工复核。"
         "先判断这张表的一行记录什么，再比较目标字段记录谁在该活动中的什么信息。根据实际字段区分目标参与方与表中的其他参与方。"
         "就最接近的两个类别比较哪一个需要更少的额外假设。候选按编号展示，编号无排名含义。"
         "给出暂定建议，即使差距很小也选一个，并明确仍缺什么。"
         '仅输出 JSON 对象，结构为 {"option_id":"候选编号","reason":"简短中文理由","fact_ids":["原始事实编号"],"uncertainty":"仍缺的证据"}。'
         "fact_ids 只填写 facts 中已有的短编号，如 F01。引用一至四个，不要编造字段值或数值置信度。"},
        {"role": "user", "content": json.dumps({"target_field": trace["initial"]["field_name"],
         "facts": {i: {"scope": row["source_id"].split(".")[:2], "attribute": row["source_id"].split(".")[-1],
                       "value": row["value"]} for i, row in facts.items()}, "candidates": options,
         "unresolved_question": gap["missing_evidence"]}, ensure_ascii=False)}
    ]
    return messages, mapping, facts


def suggestion(raw, mapping, registry, facts):
    try:
        answer = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if not isinstance(answer, dict) or set(answer) != {"option_id", "reason", "fact_ids", "uncertainty"}:
        return None
    option = answer["option_id"]
    ids = answer["fact_ids"]
    if (not isinstance(option, str) or option not in mapping or not isinstance(ids, list)
            or not 1 <= len(ids) <= 4 or not all(isinstance(k, str) and k in facts for k in ids)
            or len(set(ids)) != len(ids)
            or not all(isinstance(answer[k], str) and answer[k].strip() for k in ("reason", "uncertainty"))):
        return None
    category = mapping[option]
    return {"category": category, "grading_level": registry[category]["grading_level"],
            "grade_source": "category_registry", "review_required": True, "method": "evidence_comparison",
            "reason": answer["reason"], "uncertainty": answer["uncertainty"],
            "citations": {facts[k]["source_id"]: facts[k]["value"] for k in ids}}


def request_body(messages):
    from smoke import ROOT, common
    campaign = common.load("review_campaign", common.R9 / "campaign.py")
    config = common.read(ROOT / "config.json")
    body = campaign.build_body(messages, config)
    body.update(thinking_budget=512, max_tokens=2048)
    return config, campaign, body


def verify_review(target, output):
    from smoke import common
    record = common.read(output / "result.json")
    messages, mapping, registry, facts, path = prepare(target)
    config, campaign, body = request_body(messages)
    capture = output / "capture"
    request = common.read(output / "request.json")
    if (request["classifier_request"]["sha256"] != common.binding(path)["sha256"] or record["model"] != config["model"]
            or request["body"] != body or (capture / "request-body.bin").read_bytes() != campaign.canon(body)):
        raise ValueError("Review source or provider request differs")
    state = {}
    parser = common.load("review_sse_parser", campaign.PARSER)
    try:
        parser.parse_sse((capture / "response.sse").read_bytes().splitlines(), config["model"], state=state)
    except parser.CompletionError:
        pass
    charged = common.load("replay_review_charge", common.R9 / "recovery_policy.py").charge(state, request["reservation_CNY"])
    captured = common.read(capture / "result.json")
    complete = bool(captured.get("transport_complete") and state.get("complete") and charged["unknown"] == 0)
    if (record["wire_complete"] != complete or record["unknown_usage"] != charged["unknown"]
            or record["estimated_CNY"] != charged["conservative_CNY"]):
        raise ValueError("Review completion or accounting differs from the capture")
    if complete:
        proposed = suggestion(state.get("content"), mapping, registry, facts)
    else:
        proposed = None
    if proposed != record["suggestion"]:
        raise ValueError("Review suggestion differs from the provider response")
    return record


def run(target, output, remaining_cost, timeout, *, reserved_directory=False):
    from smoke import ROOT, common, wire
    try:
        messages, mapping, registry, facts, path = prepare(target)
    except ReviewUnavailable:
        return None
    config, campaign, body = request_body(messages)
    raw = campaign.canon(body)
    budget = campaign.Budget(1, remaining_cost, time.time() + timeout)
    try:
        reservation = budget.admit(len(raw), body["max_tokens"])
    except ValueError:
        return None
    if not reserved_directory:
        output.mkdir(parents=True, exist_ok=False)
    common.put(output / "request.json", {"classifier_request": common.binding(path), "body": body,
                                        "reservation_CNY": reservation})
    with wire.isolation.lease(ROOT / "active-worker.lock"):
        captured = wire.capture_sse(output / "capture", config["endpoint"] + "/chat/completions",
                                    raw, os.environ["DASHSCOPE_API_KEY"], config["model"], timeout)
    charge = common.load("review_accounting", common.R9 / "recovery_policy.py").charge(captured, reservation)
    valid = (captured.get("transport_complete") and captured.get("complete") and captured.get("stream_done")
             and captured.get("finish_reason") == "stop" and charge["unknown"] == 0)
    proposed = suggestion(captured.get("content"), mapping, registry, facts) if valid else None
    record = {"suggestion": proposed, "http_calls": 1, "model": config["model"],
              "source_classifier_sha256": common.binding(path)["sha256"], "wire_complete": bool(valid),
              "unknown_usage": charge["unknown"], "estimated_CNY": charge["conservative_CNY"]}
    record["review_status"] = review_status(record)
    common.put(output / "result.json", record)
    return verify_review(target, output)


if __name__ == "__main__":
    from smoke import apply_review, common, read_decision, verify
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("source", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    passed = verify(source)
    result = read_decision(source, passed)
    if not os.environ.get("DASHSCOPE_API_KEY") or result["prediction_source"] != "best_candidate":
        cli.error("A complete verified source and a Bailian credential are required")
    output.mkdir(parents=True, exist_ok=False)
    common.put(output / "decision.json", result)
    report = run(source, output, 1.0, 180, reserved_directory=True)
    if report:
        result = apply_review(result, report)
        staged = output / "decision.reviewed.partial"
        common.put(staged, result)
        os.replace(staged, output / "decision.json")
    print(json.dumps(result, ensure_ascii=True))
    sys.exit(1 if report and report["unknown_usage"] else 0)
