---
name: GET_CONTEXT_FIELD_VALUES
dsl: metadata_acquisition
required_inputs: [table_name, context_fields]
optional_inputs: []
produces: [context_field_values]
terminal: false
---
# GET_CONTEXT_FIELD_VALUES

Purpose: Retrieves available values of contextual fields and summarizes their semantic patterns.

Observation: 输出样例值模式的自然语言摘要；避免输出不必要的敏感原始值，重点描述模式和语义。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["context_field_values"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
