---
name: GET_CONTEXT_FIELD_DESCRIPTIONS
dsl: metadata_acquisition
required_inputs: [table_name, context_fields]
optional_inputs: []
produces: [context_field_descriptions]
terminal: false
---
# GET_CONTEXT_FIELD_DESCRIPTIONS

Purpose: Retrieves descriptions for selected contextual fields.

Observation: 输出所选上下文字段的描述及自然语言摘要；缺失描述需明确标记。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["context_field_descriptions"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
