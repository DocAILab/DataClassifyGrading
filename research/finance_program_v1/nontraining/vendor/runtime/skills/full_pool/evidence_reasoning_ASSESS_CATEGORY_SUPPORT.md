---
name: ASSESS_CATEGORY_SUPPORT
dsl: evidence_reasoning
required_inputs: [candidate_category, evidence]
optional_inputs: []
produces: [category_support]
terminal: false
---
# ASSESS_CATEGORY_SUPPORT

Purpose: Assesses how strongly the current evidence supports a candidate category.

Observation: 输出自然语言支持评估：支持证据、反对证据、薄弱点和总体判断。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["category_support"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
