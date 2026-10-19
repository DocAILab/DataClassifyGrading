---
name: INTERPRET_TABLE_NAME
dsl: semantic_interpretation
required_inputs: [table_name]
optional_inputs: [domain, table_description]
produces: [table_semantics]
terminal: false
---
# INTERPRET_TABLE_NAME

Purpose: Interprets the business semantics represented by a table name.

Observation: 输出表级自然语言语义，强调该表可能描述什么业务数据。

Interpret only supplied evidence. Separate direct evidence, auxiliary evidence, and uncertain hypotheses. Do not retrieve standards or predict category/grade. Never turn inferred semantics into raw metadata.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["table_semantics"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
