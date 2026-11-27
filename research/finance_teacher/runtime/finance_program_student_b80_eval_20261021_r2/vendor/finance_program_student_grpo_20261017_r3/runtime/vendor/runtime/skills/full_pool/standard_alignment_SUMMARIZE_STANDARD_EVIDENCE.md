---
name: SUMMARIZE_STANDARD_EVIDENCE
dsl: standard_alignment
required_inputs: [aligned_standard_candidates]
optional_inputs: []
produces: [standard_evidence_summary]
terminal: false
---
# SUMMARIZE_STANDARD_EVIDENCE

Purpose: Summarizes multiple standard-search and alignment results into a unified standard-evidence state.

Observation: 输出自然语言总结：当前标准证据主要支持哪些类别、哪些类别已被排除、还存在哪些冲突。

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["standard_evidence_summary"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
