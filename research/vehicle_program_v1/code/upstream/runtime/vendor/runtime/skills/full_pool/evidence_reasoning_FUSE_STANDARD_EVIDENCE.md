---
name: FUSE_STANDARD_EVIDENCE
dsl: evidence_reasoning
required_inputs: [standard_alignment_evidence]
optional_inputs: []
produces: [standard_evidence]
terminal: false
---
# FUSE_STANDARD_EVIDENCE

Purpose: Integrates multiple standard-alignment results into a unified standard-evidence state.

Observation: 输出统一标准证据状态，明确主要支持类别和弱支持类别。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["standard_evidence"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
