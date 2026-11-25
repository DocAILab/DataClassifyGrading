---
name: GET_COMPLETE_DATABASE_CONTEXT
dsl: metadata_acquisition
required_inputs: [database_name]
optional_inputs: []
produces: [database_context]
terminal: false
---
# GET_COMPLETE_DATABASE_CONTEXT

Purpose: Retrieves the available high-level metadata context of the current database.

Observation: 输出数据库级自然语言背景摘要，不引入跨库信息。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["database_context"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
