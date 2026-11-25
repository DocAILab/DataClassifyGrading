---
name: ALIGN_VALUE_TO_STANDARD
dsl: standard_alignment
required_inputs: [value_semantics, standard_candidates]
optional_inputs: []
produces: [align_value_to_standard]
terminal: false
---
# ALIGN_VALUE_TO_STANDARD

Purpose: Aligns value-derived semantics with candidate standard categories.

Observation: 输出字段值模式对标准候选的支持程度；强调值证据通常是辅助证据。

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["align_value_to_standard"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
