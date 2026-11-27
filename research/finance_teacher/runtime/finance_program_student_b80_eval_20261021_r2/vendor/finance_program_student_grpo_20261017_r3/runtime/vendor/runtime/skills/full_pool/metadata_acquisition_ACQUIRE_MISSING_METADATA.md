---
name: ACQUIRE_MISSING_METADATA
dsl: metadata_acquisition
required_inputs: [missing_metadata_type, current_metadata]
optional_inputs: []
produces: [acquired_metadata]
terminal: false
---
# ACQUIRE_MISSING_METADATA

Purpose: Acquires a specific type of missing metadata requested by the current reasoning state.

Observation: 输出尝试获取的元数据类型、实际获得的信息及其自然语言摘要；若无法获取，明确失败原因。

Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["acquired_metadata"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
