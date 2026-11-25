---
name: ALIGN_FIELD_TO_STANDARD
dsl: standard_alignment
required_inputs: [field_semantics, standard_candidates]
optional_inputs: []
produces: [field_standard_alignment]
terminal: false
---
# ALIGN_FIELD_TO_STANDARD

Purpose: Aligns the target-field semantics with candidate standard categories.

Observation: 输出字段语义与各候选标准描述的匹配/不匹配理由，以及当前更优候选。

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["field_standard_alignment"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
