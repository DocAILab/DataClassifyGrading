---
name: VERIFY_CLASSIFICATION_GRADING_CONSISTENCY
dsl: evidence_reasoning
required_inputs: [category_candidate, grading_level_candidate, current_evidence]
optional_inputs: []
produces: [grading_consistency]
terminal: false
---
# VERIFY_CLASSIFICATION_GRADING_CONSISTENCY

Purpose: Checks semantic consistency between the predicted category and Grading-level.

Observation: 输出自然语言一致性检查结论及可能的矛盾点。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["grading_consistency"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
