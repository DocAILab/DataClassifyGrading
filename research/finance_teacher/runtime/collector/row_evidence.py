"""Give the final classifier a compact, citable view of one observed table row."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from unittest.mock import patch


def identity():
    return {"version": "row-evidence-v1", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def evidence_index(facts):
    def quote(fid):
        fact = facts.get(fid, {})
        value = fact.get("value")
        if fact.get("role") == "raw_fact" and fact.get("decisive_eligible") is True and isinstance(value, str) and value.strip():
            return {"fact_id": fid, "quote": value}

    target = quote("metadata.field_description")
    table = quote("metadata.table_description")
    seen = {entry["quote"] for entry in (target, table) if entry}
    columns = []
    for fid in facts:
        if not re.fullmatch(r"observations\.(table_fields|table_context)\.fields\.\d+\.field_description", fid):
            continue
        entry = quote(fid)
        if entry and entry["quote"] not in seen:
            columns.append(entry)
            seen.add(entry["quote"])
    if columns and target:
        return {"target": target, "table": table, "other_distinct_descriptions": columns}


def project(messages):
    head = messages[0]["content"].splitlines()[0]
    if head not in {"Execute only the selected Skill INFER_CATEGORY.",
                    "Execute only the selected Skill INFER_CATEGORY_FROM_TOPK."}:
        return messages
    payload = json.loads(messages[-1]["content"])
    index = evidence_index(payload.get("evidence_boundary", {}).get("facts", {}))
    if index is None:
        return messages
    result = deepcopy(messages)
    payload["row_evidence_index"] = index
    result[-1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    result[0]["content"] += (
        "\nrow_evidence_index 把已取得的目标、表用途和其他列的原文放在一起。重复描述只列一次。"
        "先根据这些列推断一行代表的对象，再判断目标在这一行的用途。举例区分："
        "能重复使用的功能设置，与记录某一次实际操作的日志；单笔业务与主体的关系，与会计科目的汇总；"
        "主体档案，与某次活动中对该主体的引用。选最符合已见字段组合和公开定义的一项。"
        "把另一种解释视为实际竞争项前，找出本表支持它的具体列；仅仅能想象另一种用途并不构成相同强度的证据。"
        "仍须有支持所选定义必要范围的原文证据。如不同解释都有实据但无法区分，或缺少必要主体、品种或用途，则保留不足。"
        "按原合同返回结果，引用原 fact_id 和原文，不要把上述举例当作事实。")
    return result


@contextmanager
def use(runtime, record):
    if record is None:
        yield
        return
    if record != identity():
        raise ValueError("Unknown row-evidence protocol")

    class Actor(runtime.CommonActor):
        def project(self, native):
            view = super().project(native)
            shared = project(view["messages"])
            return {**view, "messages": shared, "input_sha256": runtime.previous.message_sha(shared)}

    with patch.object(runtime, "CommonActor", Actor):
        yield
