---
name: SEARCH_STANDARD_BY_CONTEXT
dsl: standard_alignment
required_inputs: [context_semantics]
optional_inputs: []
produces: [standard_candidates]
terminal: false
---
# SEARCH_STANDARD_BY_CONTEXT

Purpose: Retrieves standard categories using contextual-field semantics.

Observation: 输出上下文语义最相关的标准候选，并说明其对目标字段的辅助意义。

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["standard_candidates"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
