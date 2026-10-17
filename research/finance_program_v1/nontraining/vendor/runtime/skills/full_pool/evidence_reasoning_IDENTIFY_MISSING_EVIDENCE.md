---
name: IDENTIFY_MISSING_EVIDENCE
dsl: evidence_reasoning
required_inputs: [current_evidence, candidate_categories]
optional_inputs: []
produces: [missing_evidence]
terminal: false
---
# IDENTIFY_MISSING_EVIDENCE

Purpose: Identifies the most important missing evidence needed to discriminate among candidate categories.

Observation: 输出自然语言说明缺少哪类证据（字段描述、表字段、上下文字段语义、表描述、数据库描述、标准描述等）及其必要性。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["missing_evidence"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
