---
name: SEARCH_STANDARD
dsl: standard_alignment
required_inputs: [semantic_query]
optional_inputs: []
produces: [standard_candidates]
terminal: false
---
# Role
Retrieve ranked category candidates from the frozen standard registry.

# Evidence boundaries
Use only the supplied query and registry. Never consult evaluation labels or ground truth.

# Output
Return JSON with `summary` and `data.standard_candidates` as a ranked list of registry entries.

# Failure
Return an explicit error when the query is empty or the registry cannot be searched.
