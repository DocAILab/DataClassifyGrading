---
name: COMPARE_CATEGORY_HYPOTHESES
dsl: evidence_reasoning
required_inputs: [candidate_categories, evidence]
optional_inputs: []
produces: [category_comparison]
terminal: false
---
# COMPARE_CATEGORY_HYPOTHESES

Purpose: Compares competing standard-category hypotheses.

Observation: 输出候选类别的自然语言排序与关键区分依据。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["category_comparison"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
