---
name: FUSE_TABLE_EVIDENCE
dsl: evidence_reasoning
required_inputs: [available_table_evidence]
optional_inputs: []
produces: [table_evidence]
terminal: false
---
# FUSE_TABLE_EVIDENCE

Purpose: Integrates all available table-level semantic evidence.

Observation: 输出统一表级自然语言证据状态。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["table_evidence"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
