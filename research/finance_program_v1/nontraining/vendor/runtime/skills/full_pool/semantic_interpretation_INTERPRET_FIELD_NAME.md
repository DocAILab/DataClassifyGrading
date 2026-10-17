---
name: INTERPRET_FIELD_NAME
dsl: semantic_interpretation
required_inputs: [field_name]
optional_inputs: [domain, field_abbreviation_semantics]
produces: [field_semantics]
terminal: false
---
# Role
Interpret the likely business meaning of the supplied field identifier.

# Evidence boundaries
Use only resolved inputs and distinguish evidence from hypotheses. Do not invent unavailable table context, descriptions, values, labels, or ground truth.

# Output
Return JSON with `summary` and `data.field_semantics` as concise natural language.

# Failure
Return an explicit error when the field name is empty.
