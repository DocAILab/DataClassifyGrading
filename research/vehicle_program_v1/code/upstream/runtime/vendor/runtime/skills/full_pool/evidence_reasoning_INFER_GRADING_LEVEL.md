---
name: INFER_GRADING_LEVEL
dsl: evidence_reasoning
required_inputs: [aligned_category, aligned_standard_entry]
optional_inputs: []
produces: [grading_level]
terminal: false
---
# Role
Read the approved sensitivity level associated with the aligned registry entry.

# Evidence boundaries
Use only the aligned category and frozen registry entry. Do not infer a level from ground truth or unrelated examples.

# Output
Return JSON with `summary` and `data.grading_level`, restricted to an approved level.

# Failure
Return an explicit error when the entry does not support a valid level.
