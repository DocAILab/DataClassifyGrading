---
name: DECIDE_NEXT_EVIDENCE_SOURCE
dsl: evidence_reasoning
required_inputs: [current_evidence_state, available_metadata]
optional_inputs: []
produces: [next_evidence_source]
terminal: false
---
# DECIDE_NEXT_EVIDENCE_SOURCE

Purpose: Determines which available evidence source should be explored next.

Observation: 输出自然语言下一步建议，例如解释字段、获取同表字段、解释表/库、查询/重查标准或停止；不得调用超出可用信息范围的数据源。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["next_evidence_source"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
