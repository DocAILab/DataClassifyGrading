---
name: ALIGN_DESCRIPTION_TO_STANDARD
dsl: standard_alignment
required_inputs: [field_description_semantics, standard_candidates]
optional_inputs: []
produces: [align_description_to_standard]
terminal: false
---
# ALIGN_DESCRIPTION_TO_STANDARD

Purpose: Aligns field-description semantics with standard category descriptions.

Observation: 输出字段描述与各标准候选的语义匹配理由，显式描述通常可作为较强证据。

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["align_description_to_standard"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
