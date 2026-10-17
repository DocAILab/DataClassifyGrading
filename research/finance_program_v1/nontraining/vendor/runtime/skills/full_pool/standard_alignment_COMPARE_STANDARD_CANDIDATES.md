---
name: COMPARE_STANDARD_CANDIDATES
dsl: standard_alignment
required_inputs: [standard_candidates, semantic_evidence]
optional_inputs: []
produces: [standard_comparison, aligned_standard_entry]
terminal: false
---
# Role
Compare retrieved candidates against the current semantic evidence and identify the best-supported entry.

# Evidence boundaries
Compare only supplied candidates. Do not add categories, hidden metadata, labels, or ground truth.

# Output
Return JSON with a nonempty `summary`. `data.standard_comparison` must be a non-empty string explaining the comparison, never a list. `data.aligned_standard_entry` must be one exact object copied from `standard_candidate_details`, containing exactly `category`, `label`, and `description`. Copy all three values without changes or extra fields. Select only an actual supplied candidate.

# Failure
Return an explicit error when candidates or semantic evidence are absent.
