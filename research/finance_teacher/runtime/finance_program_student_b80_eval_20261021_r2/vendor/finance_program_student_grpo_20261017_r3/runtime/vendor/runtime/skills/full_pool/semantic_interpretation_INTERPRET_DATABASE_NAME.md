---
name: INTERPRET_DATABASE_NAME
dsl: semantic_interpretation
required_inputs: [database_name]
optional_inputs: [domain, database_description]
produces: [database_semantics]
terminal: false
---
# INTERPRET_DATABASE_NAME

Purpose: Interprets the domain and semantic scope implied by the database name.

Observation: 输出数据库级自然语言语义，作为当前字段的高层背景。

Interpret only supplied evidence. Separate direct evidence, auxiliary evidence, and uncertain hypotheses. Do not retrieve standards or predict category/grade. Never turn inferred semantics into raw metadata.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["database_semantics"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
