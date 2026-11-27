---
name: SYNTHESIZE_METADATA_SEMANTICS
dsl: semantic_interpretation
required_inputs: [field_semantics]
optional_inputs: [domain_semantics, table_semantics, database_semantics, context_field_semantics, field_description_semantics, field_type_semantics, value_semantics]
produces: [metadata_semantics]
terminal: false
---
# SYNTHESIZE_METADATA_SEMANTICS

Purpose: Synthesizes all available metadata interpretations into a unified natural-language semantic state.

Observation: 输出一段紧凑但信息充分的自然语言 semantic state，区分强证据、辅助证据和仍存在的歧义。

Interpret only supplied evidence. Separate direct evidence, auxiliary evidence, and uncertain hypotheses. Do not retrieve standards or predict category/grade. Never turn inferred semantics into raw metadata.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["metadata_semantics"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
