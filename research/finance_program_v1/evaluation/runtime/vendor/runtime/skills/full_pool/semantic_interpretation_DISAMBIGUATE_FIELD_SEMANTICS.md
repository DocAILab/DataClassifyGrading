---
name: DISAMBIGUATE_FIELD_SEMANTICS
dsl: semantic_interpretation
required_inputs: [field_semantics]
optional_inputs: [domain, standard_candidates]
produces: [disambiguated_field_semantics]
terminal: false
---
# Role
Resolve competing interpretations using only the current semantic state and supplied candidates.

# Evidence boundaries
Preserve uncertainty when evidence is insufficient. Do not use hidden metadata, labels, or ground truth.

# Output
Return JSON with `summary` and `data.disambiguated_field_semantics` as concise natural language.

# Failure
Return an explicit error when no field semantics are available.
