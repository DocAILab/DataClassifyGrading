---
name: RESOLVE_METADATA_CONFLICT
dsl: evidence_reasoning
required_inputs: [conflicting_metadata_evidence]
optional_inputs: []
produces: [metadata_conflict_resolution]
terminal: false
---
# RESOLVE_METADATA_CONFLICT

Purpose: Resolves conflicting semantic evidence across metadata granularities.

Observation: 输出自然语言冲突消解结论和依据。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["metadata_conflict_resolution"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
