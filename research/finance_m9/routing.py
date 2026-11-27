def adapter_route(stage):
    if stage == "Planner":
        return "planner"
    if stage in {"INFER_CATEGORY", "INFER_CATEGORY_FROM_TOPK"}:
        return "skill_fixed"
    return None
