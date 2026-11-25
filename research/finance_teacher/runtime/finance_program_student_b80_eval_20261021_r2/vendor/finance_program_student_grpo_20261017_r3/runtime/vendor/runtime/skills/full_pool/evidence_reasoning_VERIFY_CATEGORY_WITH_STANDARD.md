---
name: VERIFY_CATEGORY_WITH_STANDARD
dsl: evidence_reasoning
required_inputs: [candidate_category, standard_evidence, metadata_evidence]
optional_inputs: []
produces: [category_verification]
terminal: false
---
# VERIFY_CATEGORY_WITH_STANDARD

Purpose: Verifies whether a candidate category is jointly supported by metadata and standard evidence.

Observation: 输出自然语言验证结论，明确标准描述与元数据语义是否一致。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["category_verification"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
