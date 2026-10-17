---
name: INTERPRET_FIELD_DESCRIPTION
dsl: semantic_interpretation
required_inputs: [field_description]
optional_inputs: [field_name, domain]
produces: [field_description_semantics]
terminal: false
---
# Role
Explain the business meaning of an observed field description, without assigning a category.

# Evidence boundary
Use only the supplied real description and optional supplied context. Separate stated facts
from uncertain interpretation. Never invent values, neighboring fields, labels or grading levels.

# Output
Return JSON with a short `summary` and `data.field_description_semantics` as concise text.

# Failure
Do not infer from an absent or empty field description.
