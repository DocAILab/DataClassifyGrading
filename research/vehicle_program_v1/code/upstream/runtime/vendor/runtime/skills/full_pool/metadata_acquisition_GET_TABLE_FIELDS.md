---
name: GET_TABLE_FIELDS
dsl: metadata_acquisition
required_inputs: [table_name]
optional_inputs: []
produces: [table_fields]
terminal: false
---
# GET_TABLE_FIELDS

Purpose: Retrieves other fields from the current table to provide local schema context.

Observation: 输出同表字段名称列表的自然语言摘要，突出可能与目标字段相关的字段；不臆造不存在字段。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["table_fields"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
