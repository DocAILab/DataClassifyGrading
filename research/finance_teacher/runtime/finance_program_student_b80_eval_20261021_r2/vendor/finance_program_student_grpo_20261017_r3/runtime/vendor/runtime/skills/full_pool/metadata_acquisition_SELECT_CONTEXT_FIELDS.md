---
name: SELECT_CONTEXT_FIELDS
dsl: metadata_acquisition
required_inputs: [target_field, table_fields]
optional_inputs: [current_semantic_state]
produces: [selected_context_fields]
terminal: false
---
# SELECT_CONTEXT_FIELDS

Purpose: Selects the most informative contextual fields for the target field.

Observation: 输出自然语言说明选择了哪些上下文字段、为什么它们更有信息价值，以及哪些字段被忽略。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["selected_context_fields"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
