---
name: WEIGH_EVIDENCE
dsl: evidence_reasoning
required_inputs: [evidence_set]
optional_inputs: []
produces: [evidence_weights]
terminal: false
---
# WEIGH_EVIDENCE

Purpose: Assesses the relative reliability and importance of available evidence.

Observation: 输出自然语言说明哪些证据更强、哪些只是辅助，以及原因；不输出未经依据的数值权重。

Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["evidence_weights"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
