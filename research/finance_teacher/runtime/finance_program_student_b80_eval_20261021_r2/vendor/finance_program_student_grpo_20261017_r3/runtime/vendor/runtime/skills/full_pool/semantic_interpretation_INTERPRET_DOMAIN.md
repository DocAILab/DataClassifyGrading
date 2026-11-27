---
name: INTERPRET_DOMAIN
dsl: semantic_interpretation
required_inputs: [domain]
optional_inputs: []
produces: [domain_semantics]
terminal: false
---
# Role
Describe the semantic scope implied by the supplied domain.

# Evidence boundaries
Use only the domain string. Do not replace it with a dataset name or infer hidden metadata or ground truth.

# Output
Return JSON with `summary` and `data.domain_semantics` as concise natural language.

# Failure
Return an explicit error when the domain is empty.
