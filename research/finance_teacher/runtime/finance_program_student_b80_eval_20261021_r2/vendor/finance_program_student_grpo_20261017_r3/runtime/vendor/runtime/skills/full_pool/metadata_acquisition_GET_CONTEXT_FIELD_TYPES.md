---
name: GET_CONTEXT_FIELD_TYPES
dsl: metadata_acquisition
required_inputs: [table_name, context_fields]
optional_inputs: []
produces: [context_field_types]
terminal: false
---
# GET_CONTEXT_FIELD_TYPES

Purpose: Retrieves data types of selected contextual fields as supporting evidence.

Observation: 输出上下文字段类型的自然语言摘要，说明类型能提供的有限辅助信息。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["context_field_types"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
