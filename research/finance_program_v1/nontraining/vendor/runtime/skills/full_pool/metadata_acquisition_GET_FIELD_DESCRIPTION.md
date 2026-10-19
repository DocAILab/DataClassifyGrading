---
name: GET_FIELD_DESCRIPTION
dsl: metadata_acquisition
required_inputs: [table_name, field_name]
optional_inputs: []
produces: [field_description]
terminal: false
---
# GET_FIELD_DESCRIPTION

Purpose: Retrieves the description of the target field from the available database metadata.

Observation: 输出获取到的字段描述及自然语言摘要；若不存在，应明确说明未获得该信息。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["field_description"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
