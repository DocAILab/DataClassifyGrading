---
name: INTERPRET_CONTEXT_FIELDS
dsl: semantic_interpretation
required_inputs: [context_fields]
optional_inputs: [domain, table_name]
produces: [context_field_semantics]
terminal: false
---
# INTERPRET_CONTEXT_FIELDS

Purpose: Interprets neighboring fields and summarizes the business context jointly implied by them.

Observation: 输出各上下文字段的简要语义及它们共同指向的业务场景；不直接替目标字段做最终分类。

Interpret only supplied evidence. Separate direct evidence, auxiliary evidence, and uncertain hypotheses. Do not retrieve standards or predict category/grade. Never turn inferred semantics into raw metadata.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["context_field_semantics"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
