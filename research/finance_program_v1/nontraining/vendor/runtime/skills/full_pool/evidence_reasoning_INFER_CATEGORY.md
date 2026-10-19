---
name: INFER_CATEGORY
dsl: evidence_reasoning
required_inputs: [candidate_standard_categories, all_available_evidence]
optional_inputs: []
produces: [category, aligned_standard_entry]
terminal: false
---
# Role
Select the best-supported category from the supplied registry candidates.

# Evidence boundaries
The category and aligned entry must come from the supplied candidates. Do not invent categories or use ground truth.

# Output
Return JSON with `summary`, `data.category`, and `data.aligned_standard_entry`.

# Failure
Return an explicit error when no candidate is supported.
