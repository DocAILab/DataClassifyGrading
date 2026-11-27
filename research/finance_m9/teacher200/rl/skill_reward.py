"""Score only native, completed TRAIN episodes after the Planner has finished."""


def skill_reward(row, trace, reference, metadata):
    if row["source_id"] != trace["source_id"] or row["prediction"] != trace["final_result"]:
        raise ValueError("Trace and rollout receipt disagree")
    if not row["valid"] or trace["stop_reason"] != "validated_terminal":
        return dict(outcome=0, retrieval=0., evidence=0., total=0.)
    outcome = int(row["prediction"] == reference)
    steps = [s for p in trace["programs"] for s in p["steps"]]
    infer = None
    candidates = []
    for step in reversed(steps):
        if step["status"] != "success":
            continue
        tool = step["call"]["tool"]
        if infer is None:
            if tool in ("INFER_CATEGORY", "INFER_CATEGORY_FROM_TOPK"):
                infer = step
        elif tool.startswith("SEARCH_STANDARD_"):
            candidates = [c.get("category") for c in step["data"].get("standard_candidates", [])[:5]]
            break
    retrieval = evidence = 0.
    if infer and infer["data"].get("category") == row["prediction"]["category"]:
        if reference["category"] in candidates:
            retrieval = .12 / (1 + candidates.index(reference["category"]))
        for audit in infer.get("evidence_audits", []):
            support = audit.get("support") or {}
            if audit.get("status") != "validated_citations" or not audit.get("backend_called") or not support.get("sufficient"):
                continue
            matched = set()
            for cite in support.get("citations", []):
                fact = cite.get("fact_id", "")
                quote = (cite.get("quote") or "").strip()
                for name in ("field_description", "table_description"):
                    if fact == "metadata."+name and len(quote) >= 2 and quote in metadata.get(name, ""):
                        matched.add(name)
            if len(matched) == 2:
                evidence = .02
                break
    return dict(outcome=outcome, retrieval=retrieval, evidence=evidence, total=outcome+retrieval+evidence)
