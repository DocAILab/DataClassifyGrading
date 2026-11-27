---
name: FUSE_METADATA_EVIDENCE
dsl: evidence_reasoning
required_inputs: [available_metadata_evidence]
optional_inputs: []
produces: [metadata_evidence]
terminal: false
---
# FUSE_METADATA_EVIDENCE

Purpose: Integrates multi-granularity metadata evidence into a unified semantic state.

Observation: 输出统一 metadata evidence state，并指出不同粒度证据的一致与冲突。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["metadata_evidence"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
