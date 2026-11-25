---
name: REFINE_STANDARD_QUERY
dsl: standard_alignment
required_inputs: [previous_standard_results, current_semantic_state]
optional_inputs: []
produces: [refined_semantic_query]
terminal: false
---
# REFINE_STANDARD_QUERY

Purpose: Refines the standard-search query using previous retrieval results and the current semantic state.

Observation: 输出一段更聚焦的自然语言查询表达，并说明为何这样改写能区分当前候选。

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["refined_semantic_query"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
