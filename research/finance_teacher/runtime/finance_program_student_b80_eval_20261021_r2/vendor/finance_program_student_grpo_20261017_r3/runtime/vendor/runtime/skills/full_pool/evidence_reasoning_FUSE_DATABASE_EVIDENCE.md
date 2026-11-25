---
name: FUSE_DATABASE_EVIDENCE
dsl: evidence_reasoning
required_inputs: [available_database_evidence]
optional_inputs: []
produces: [database_evidence]
terminal: false
---
# FUSE_DATABASE_EVIDENCE

Purpose: Integrates database-level semantic evidence.

Observation: 输出数据库级自然语言背景及其对目标字段的高层约束。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["database_evidence"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
