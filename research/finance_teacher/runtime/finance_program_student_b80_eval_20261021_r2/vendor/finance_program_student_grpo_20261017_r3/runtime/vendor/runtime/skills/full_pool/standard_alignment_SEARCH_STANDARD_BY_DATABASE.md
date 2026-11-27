---
name: SEARCH_STANDARD_BY_DATABASE
dsl: standard_alignment
required_inputs: [database_semantics]
optional_inputs: []
produces: [standard_candidates]
terminal: false
---
# SEARCH_STANDARD_BY_DATABASE

Purpose: Uses database-level semantics to retrieve or constrain candidate standard categories.

Observation: 输出数据库级语义可能支持的标准类别范围和依据。

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["standard_candidates"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
