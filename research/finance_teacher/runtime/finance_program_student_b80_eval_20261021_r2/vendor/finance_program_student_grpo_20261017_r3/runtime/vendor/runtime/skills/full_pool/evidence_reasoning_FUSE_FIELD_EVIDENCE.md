---
name: FUSE_FIELD_EVIDENCE
dsl: evidence_reasoning
required_inputs: [available_field_evidence]
optional_inputs: []
produces: [field_evidence]
terminal: false
---
# FUSE_FIELD_EVIDENCE

Purpose: Integrates all available evidence directly associated with the target field.

Observation: 输出字段级统一自然语言证据状态，区分直接证据、辅助证据和歧义。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["field_evidence"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
