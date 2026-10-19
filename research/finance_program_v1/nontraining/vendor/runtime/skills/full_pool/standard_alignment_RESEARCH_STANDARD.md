---
name: RESEARCH_STANDARD
dsl: standard_alignment
required_inputs: [refined_semantic_query]
optional_inputs: []
produces: [standard_candidates]
terminal: false
---
# RESEARCH_STANDARD

Purpose: Performs a subsequent standard search after semantic refinement.

Observation: 输出新一轮标准候选、与上一轮相比新增或变化的证据，以及是否更有区分度。

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["standard_candidates"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
