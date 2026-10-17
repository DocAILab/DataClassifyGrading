---
name: EXPAND_TABLE_ABBREVIATION
dsl: semantic_interpretation
required_inputs: [table_name]
optional_inputs: [domain, table_description]
produces: [table_abbreviation_semantics]
terminal: false
---
# EXPAND_TABLE_ABBREVIATION

Purpose: Expands an abbreviated table name into plausible business meanings.

Observation: 输出表名缩写的候选展开，以及它可能表示的数据主题或业务对象。

Interpret only supplied evidence. Separate direct evidence, auxiliary evidence, and uncertain hypotheses. Do not retrieve standards or predict category/grade. Never turn inferred semantics into raw metadata.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["table_abbreviation_semantics"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
