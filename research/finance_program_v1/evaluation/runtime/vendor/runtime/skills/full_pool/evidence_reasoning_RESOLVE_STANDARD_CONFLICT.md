---
name: RESOLVE_STANDARD_CONFLICT
dsl: evidence_reasoning
required_inputs: [conflicting_standard_alignments]
optional_inputs: []
produces: [standard_conflict_resolution]
terminal: false
---
# RESOLVE_STANDARD_CONFLICT

Purpose: Resolves conflicts among competing standard alignments.

Observation: 输出哪个标准候选更可信、为什么，以及仍存在的歧义。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["standard_conflict_resolution"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
