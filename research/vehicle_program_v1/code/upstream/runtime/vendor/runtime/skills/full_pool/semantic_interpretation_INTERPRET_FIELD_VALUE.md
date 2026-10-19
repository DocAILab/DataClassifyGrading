---
name: INTERPRET_FIELD_VALUE
dsl: semantic_interpretation
required_inputs: [value]
optional_inputs: [field_name, field_type, domain]
produces: [value_semantics]
terminal: false
---
# INTERPRET_FIELD_VALUE

Purpose: Infers possible semantics from the observed value pattern and content.

Observation: 输出值模式的自然语言描述、可能对应的数据含义，以及这种证据的可靠性边界。

Interpret only supplied evidence. Separate direct evidence, auxiliary evidence, and uncertain hypotheses. Do not retrieve standards or predict category/grade. Never turn inferred semantics into raw metadata.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["value_semantics"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
