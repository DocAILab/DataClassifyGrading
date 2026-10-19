---
name: GET_COMPLETE_TABLE_CONTEXT
dsl: metadata_acquisition
required_inputs: [table_name]
optional_inputs: []
produces: [table_context]
terminal: false
---
# GET_COMPLETE_TABLE_CONTEXT

Purpose: Retrieves the complete available metadata context of the current table.

Observation: 输出当前表完整可用元数据的自然语言摘要，并明确哪些属性缺失。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["table_context"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
