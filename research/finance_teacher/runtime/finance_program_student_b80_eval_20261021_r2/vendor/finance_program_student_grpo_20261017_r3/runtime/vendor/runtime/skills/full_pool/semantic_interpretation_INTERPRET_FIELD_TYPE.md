---
name: INTERPRET_FIELD_TYPE
dsl: semantic_interpretation
required_inputs: [field_type]
optional_inputs: [field_name, field_description]
produces: [field_type_semantics]
terminal: false
---
# INTERPRET_FIELD_TYPE

Purpose: Uses the field data type as supporting semantic evidence.

Observation: 输出字段类型可提供的约束性自然语言判断；不得仅凭类型强行确定具体类别。

Interpret only supplied evidence. Separate direct evidence, auxiliary evidence, and uncertain hypotheses. Do not retrieve standards or predict category/grade. Never turn inferred semantics into raw metadata.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["field_type_semantics"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
