---
name: FINALIZE_CLASSIFICATION
dsl: evidence_reasoning
required_inputs: [final_evidence_state]
optional_inputs: []
produces: [final_result]
terminal: true
---
# Role
Return the final category and grading level supported by the current evidence state.

# Evidence boundaries
Use existing evidence only. The category must exist in the frozen registry and the level must be approved. Never use ground truth.

# Output
Return exactly one JSON object with `summary` and `data.final_result`, where the result contains both `category` and `grading_level`.

# Failure
Return an explicit error when a valid complete result cannot be produced.
