"""Produce one final category and grade for each verified business outcome."""
import json
import re


STOP = "unresolved_no_new_business_evidence"


def verified_abstention(trace, proof):
    """An earlier recoverable formatting error does not erase the final native stop."""
    rows = proof.get("rows", [])
    if len(rows) != 1:
        return False
    row, = rows
    summary = proof["summary"]
    return (not summary.get("TestOnly") and summary["passed"] and summary["actual_OS_exit"] == 0 and summary["completed"] == 1
            and summary["native_replays"] == 1 and summary["wire_complete_all"]
            and summary["same_information_exact"] and summary["unknown_usage"] == 0
            and row["native_replay_exact"] and not row["legal"] and not trace.get("terminated")
            and row["source_id"] == trace.get("source_id") and row["stop_reason"] == trace.get("stop_reason") == STOP)


def terms(value):
    words = set(re.findall(r"[a-z0-9]+", value.lower()))
    for chunk in re.findall(r"[\u3400-\u9fff]+", value):
        for size in (2, 3):
            words.update(chunk[i:i + size] for i in range(len(chunk) - size + 1))
    return words


def catalog_choice(trace, registry):
    """Rank public category definitions against observed metadata, exposing ties."""
    state = trace.get("state", {})
    active = {c["category"] for c in (state.get("standard_candidates") or [])
              if isinstance(c, dict) and isinstance(c.get("category"), str) and c["category"] in registry}
    pool = sorted(active or registry.keys())
    if not pool:
        raise ValueError("The category registry must contain at least one candidate")
    gaps = trace.get("gap_ledger") or [{}]
    facts = {k: v for k, v in gaps[-1].get("checkpoint", {}).items() if isinstance(v, str) and v.strip()}
    for key in ("field_description", "table_description", "field_name"):
        value = state.get(key) or trace.get("initial", {}).get(key)
        if isinstance(value, str) and value.strip():
            facts.setdefault("metadata." + key, value)
    observed = {key: terms(value) for key, value in facts.items()}
    target_terms = observed.get("metadata.field_description", set())
    all_terms = set().union(*observed.values())
    scores = {}
    for category in pool:
        entry = registry[category]
        document = " ".join((category.split(":", 1)[-1], entry.get("description", "")))
        vocabulary = terms(document)
        scores[category] = (len(target_terms & vocabulary), len(all_terms & vocabulary))
    highest = max(scores.values())
    ties = [c for c in pool if scores[c] == highest]
    selected = ties[0]
    vocabulary = terms(selected.split(":", 1)[-1] + " " + registry[selected].get("description", ""))
    matched = sorted(all_terms & vocabulary, key=lambda term: (-len(term), term))
    method = "catalog_text"
    reason = "按目标字段说明及已取得的元数据与分类标准的文字匹配排序。"
    if len(pool) == 1:
        method, reason = "single_candidate", "当前有效候选只剩这一项。"
    elif len(ties) > 1:
        method, reason = "catalog_tie", "现有字段信息无法区分并列首位的候选，按稳定的类别标识顺序展示其中一个。"
    return {"category": selected, "grading_level": registry[selected]["grading_level"],
            "method": method, "matched_terms": matched[:6],
            "rank": 1, "grade_source": "category_registry", "review_required": True,
            "tied_candidates": ties if len(ties) > 1 else [],
            "reason": reason}


def select_candidate(result, candidate):
    """Promote the best available choice and retain its rationale for audit."""
    prediction = {key: candidate[key] for key in ("category", "grading_level")}
    return {**result, "status": "classified", "prediction": prediction, "prediction_source": "best_candidate",
            "selection_detail": {k: v for k, v in candidate.items() if k not in {"category", "grading_level", "review_required"}}}


def review_status(report):
    if report["suggestion"]:
        return "accepted"
    return "invalid_suggestion" if report["wire_complete"] else "incomplete_response"


def apply_review(result, review):
    if result["prediction_source"] != "best_candidate":
        raise ValueError("Only an evidence-limited classification can use a candidate review")
    if review["suggestion"] is not None:
        result = select_candidate(result, review["suggestion"])
    return {**result, "review_status": review_status(review),
            "review_accounting": {k: review[k] for k in ("http_calls", "estimated_CNY", "unknown_usage")}}


def decision(trace, registry, verified):
    reason = trace.get("stop_reason")
    result = {"status": "incomplete", "stop_reason": reason, "prediction": None,
              "prediction_source": None, "missing_evidence": None}
    if not verified:
        return result
    if reason == "validated_terminal" and trace.get("terminated") and trace.get("final_result"):
        return {**result, "status": "classified", "prediction": trace["final_result"], "prediction_source": "evidence"}
    if reason != STOP or trace.get("terminated"):
        return result

    result["review_status"] = "not_run"
    gaps = trace.get("gap_ledger", [])
    if gaps:
        result["missing_evidence"] = gaps[-1].get("missing_evidence")
    candidate = ranked_choice(trace, registry) or catalog_choice(trace, registry)
    return select_candidate(result, candidate)


def ranked_choice(trace, registry):
    state = trace.get("state", {})
    active = state.get("standard_candidates")
    producer = state.get("_provenance", {}).get("standard_candidates", "")
    if (not active or not isinstance(producer, str)
            or not (producer in {"SELECT_CATEGORY_TOPK", "RESEARCH_STANDARD"} or producer.startswith("SEARCH_STANDARD"))):
        return None
    lineage = trace.get("final_lineage", {}).get("candidates")
    if lineage is None:
        return None
    for program in reversed(trace.get("programs", [])):
        for step in reversed(program["steps"]):
            if (step["status"] != "success" or step["call"]["tool"] != producer
                    or step["lineage"].get("candidates") != lineage
                    or step["data"].get("standard_candidates") != active):
                continue
            # Require the native model ranking. A displayed catalog may be sorted by ID.
            metadata = step.get("metadata", {})
            raw = metadata.get("raw_response")
            messages = metadata.get("messages")
            if not isinstance(raw, str) or not messages:
                return None
            try:
                ranking = json.loads(raw)
                catalog = json.loads(messages[-1]["content"])["category_options"]
                labels = {option["option_id"]: option["label"] for option in catalog}
            except (ValueError, KeyError, TypeError, IndexError):
                return None
            ids = ranking.get("option_ids") if isinstance(ranking, dict) else None
            if (not isinstance(ids, list) or len(ids) != len(active)
                    or not all(isinstance(i, str) and i for i in ids) or len(set(ids)) != len(ids)):
                return None
            if not all(isinstance(c, dict) and isinstance(c.get("label"), str) for c in active):
                return None
            if [labels.get(i) for i in ids] != [candidate["label"] for candidate in active]:
                return None
            category = active[0].get("category")
            if not isinstance(category, str) or category not in registry:
                return None
            return {"category": category, "grading_level": registry[category]["grading_level"],
                                    "method": "retrieval_rank", "rank": 1, "ranking_skill": producer, "ranking_summary": step["summary"],
                                    "grade_source": "category_registry", "review_required": True}
    return None
