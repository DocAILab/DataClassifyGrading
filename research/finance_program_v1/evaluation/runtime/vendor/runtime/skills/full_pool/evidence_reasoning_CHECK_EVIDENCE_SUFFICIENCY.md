---
name: CHECK_EVIDENCE_SUFFICIENCY
dsl: evidence_reasoning
required_inputs: [current_evidence]
optional_inputs: []
produces: [evidence_sufficiency]
terminal: false
---
# Role
Judge whether the current evidence supports category and grading decisions.

# Evidence boundaries
Use only the supplied evidence state. Missing evidence must remain missing; do not infer ground truth.

# Output
Return JSON with `summary` and `data.evidence_sufficiency`, including a boolean `sufficient` and concise rationale.

# Failure
Return an explicit error when no evidence is available.
