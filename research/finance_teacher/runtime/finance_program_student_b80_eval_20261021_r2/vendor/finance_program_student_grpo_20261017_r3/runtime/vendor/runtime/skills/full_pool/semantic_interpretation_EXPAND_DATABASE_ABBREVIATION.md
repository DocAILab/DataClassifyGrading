---
name: EXPAND_DATABASE_ABBREVIATION
dsl: semantic_interpretation
required_inputs: [database_name]
optional_inputs: [domain, database_description]
produces: [database_abbreviation_semantics]
terminal: false
---
# EXPAND_DATABASE_ABBREVIATION

Purpose: Expands an abbreviated database or system name into its likely semantic meaning.

Observation: 输出数据库/系统缩写的自然语言展开及可能的业务范围。

Interpret only supplied evidence. Separate direct evidence, auxiliary evidence, and uncertain hypotheses. Do not retrieve standards or predict category/grade. Never turn inferred semantics into raw metadata.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["database_abbreviation_semantics"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
