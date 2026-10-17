---
name: INTERPRET_DATABASE_DESCRIPTION
dsl: semantic_interpretation
required_inputs: [database_description]
optional_inputs: [database_name, domain]
produces: [database_description_semantics]
terminal: false
---
# INTERPRET_DATABASE_DESCRIPTION

Purpose: Interprets the semantic scope expressed by the database description.

Observation: 输出数据库描述的自然语言摘要，并说明其对字段判断的约束。

Interpret only supplied evidence. Separate direct evidence, auxiliary evidence, and uncertain hypotheses. Do not retrieve standards or predict category/grade. Never turn inferred semantics into raw metadata.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["database_description_semantics"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
