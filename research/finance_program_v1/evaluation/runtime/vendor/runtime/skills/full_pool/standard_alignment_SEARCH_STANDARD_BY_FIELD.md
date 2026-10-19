---
name: SEARCH_STANDARD_BY_FIELD
dsl: standard_alignment
required_inputs: [field_semantics]
optional_inputs: []
produces: [standard_candidates]
terminal: false
---
# SEARCH_STANDARD_BY_FIELD

Purpose: Retrieves standard categories relevant to the target-field semantics.

Observation: 输出字段语义最相关的标准候选及匹配依据。

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["standard_candidates"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
