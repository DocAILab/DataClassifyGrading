---
name: FUSE_CONTEXT_EVIDENCE
dsl: evidence_reasoning
required_inputs: [context_field_evidence]
optional_inputs: []
produces: [context_evidence]
terminal: false
---
# FUSE_CONTEXT_EVIDENCE

Purpose: Integrates semantic evidence from selected contextual fields.

Observation: 输出自然语言总结上下文字段共同指向的业务场景及其对目标字段的支持。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["context_evidence"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
