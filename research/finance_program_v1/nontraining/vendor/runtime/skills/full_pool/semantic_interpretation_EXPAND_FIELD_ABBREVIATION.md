---
name: EXPAND_FIELD_ABBREVIATION
dsl: semantic_interpretation
required_inputs: [field_name]
optional_inputs: [domain]
produces: [field_abbreviation_semantics]
terminal: false
---
# Role
Expand identifier tokens and likely abbreviations in the supplied field name.

# Evidence boundaries
Mark uncertain expansions explicitly. Do not invent table context, values, descriptions, labels, or ground truth.

# Output
Return JSON with `summary` and `data.field_abbreviation_semantics` as concise natural language.

# Failure
Return an explicit error when the field name is empty.
