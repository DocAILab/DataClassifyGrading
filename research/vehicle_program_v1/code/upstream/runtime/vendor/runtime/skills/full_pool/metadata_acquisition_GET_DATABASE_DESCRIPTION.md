---
name: GET_DATABASE_DESCRIPTION
dsl: metadata_acquisition
required_inputs: [database_name]
optional_inputs: []
produces: [database_description]
terminal: false
---
# GET_DATABASE_DESCRIPTION

Purpose: Retrieves and summarizes the description of the current database.

Observation: 输出数据库描述及自然语言摘要；不存在时明确说明。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["database_description"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
