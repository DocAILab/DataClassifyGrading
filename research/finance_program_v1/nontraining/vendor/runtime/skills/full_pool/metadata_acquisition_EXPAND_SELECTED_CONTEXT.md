---
name: EXPAND_SELECTED_CONTEXT
dsl: metadata_acquisition
required_inputs: [selected_context_fields]
optional_inputs: []
produces: [context_fields]
terminal: false
---
# EXPAND_SELECTED_CONTEXT

Purpose: Organizes selected contextual metadata into a usable state for subsequent semantic interpretation.

Observation: 输出自然语言上下文摘要，保持字段名与其可用描述/类型/值之间的对应关系。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["context_fields"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
