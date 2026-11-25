---
name: RESOLVE_FIELD_CONTEXT_CONFLICT
dsl: evidence_reasoning
required_inputs: [field_evidence, context_evidence]
optional_inputs: []
produces: [field_context_resolution]
terminal: false
---
# RESOLVE_FIELD_CONTEXT_CONFLICT

Purpose: Resolves conflicts between target-field evidence and contextual-field evidence.

Observation: 输出冲突点、比较依据、保留的解释以及仍需补充的信息。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["field_context_resolution"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
