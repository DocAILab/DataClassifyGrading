---
name: ALIGN_CONTEXT_TO_STANDARD
dsl: standard_alignment
required_inputs: [context_semantics, standard_candidates]
optional_inputs: []
produces: [align_context_to_standard]
terminal: false
---
# ALIGN_CONTEXT_TO_STANDARD

Purpose: Aligns contextual semantic evidence with candidate standard categories.

Observation: 输出上下文语义对候选标准类别的支持、反对或弱相关关系。

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["align_context_to_standard"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
