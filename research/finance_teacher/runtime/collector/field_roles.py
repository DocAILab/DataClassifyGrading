"""Expose distinct literal qualifiers when two columns end in the same attribute."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from os.path import commonprefix
from pathlib import Path
import re
from unittest.mock import patch


def identity():
    return {"version": "distinct-field-qualifiers-v1", "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def comparisons(index):
    target = index["target"]["quote"]
    found = []
    for column in index["other_distinct_descriptions"]:
        other = column["quote"]
        suffix = commonprefix((target[::-1], other[::-1]))[::-1]
        if not re.fullmatch(r"[\u3400-\u9fff]{3,}", suffix):
            continue
        a, b = target[:-len(suffix)].strip(), other[:-len(suffix)].strip()
        if len(a) >= 2 and len(b) >= 2 and a != b:
            found.append({"target_fact_id": index["target"]["fact_id"], "other_fact_id": column["fact_id"],
                          "shared_suffix": suffix, "target_prefix": a, "other_prefix": b})
    return found


def project(messages):
    head = messages[0]["content"].splitlines()[0]
    if head not in {"Execute only the selected Skill INFER_CATEGORY.",
                    "Execute only the selected Skill INFER_CATEGORY_FROM_TOPK."}:
        return messages
    payload = json.loads(messages[-1]["content"])
    index = payload.get("row_evidence_index")
    contrasts = comparisons(index) if index else []
    if not contrasts:
        return messages
    payload["contrasting_field_wording"] = contrasts
    result = deepcopy(messages)
    result[-1]["content"] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    result[0]["content"] += (
        "\n注意 contrasting_field_wording 指向同表中属性尾词相同、前面限定语不同的两列。"
        "区分这个限定语描述的是人、时间还是业务关系，再按照目标列自己写明的对象作判断。"
        "同表可以同时记录多个主体、时期或环节。要把另一列的角色或关系套在目标列上，必须有原文把二者联系起来；"
        "多个相邻列也不能替换目标列明确写出的角色。结合目标与最接近的其他列核对定义，"
        "引用真正支持判断的原始事实。字面切分只是阅读线索，如不足以支持任何候选，继续使用原有证据不足分支。")
    return result


@contextmanager
def use(runtime, record):
    if record is None:
        yield
        return
    if record != identity():
        raise ValueError("Unknown field-qualifier protocol")

    class Actor(runtime.CommonActor):
        def project(self, native):
            view = super().project(native)
            shared = project(view["messages"])
            return {**view, "messages": shared, "input_sha256": runtime.previous.message_sha(shared)}

    with patch.object(runtime, "CommonActor", Actor):
        yield
