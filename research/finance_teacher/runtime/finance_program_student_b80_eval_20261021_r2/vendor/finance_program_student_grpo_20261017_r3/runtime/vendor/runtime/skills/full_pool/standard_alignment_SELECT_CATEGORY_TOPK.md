---
name: SELECT_CATEGORY_TOPK
dsl: standard_alignment
required_inputs: [semantic_query]
optional_inputs: []
produces: [standard_candidates]
terminal: false
---
# Role
Select ranked Top-K short option IDs from the full frozen label catalog. The executor maps each selected option back to its canonical Registry category ID.

# Evidence boundary
This stage receives deterministic short option IDs and full category-path labels only, never canonical long IDs, category descriptions, grading levels, or evaluation labels. Do not invent option IDs.

# Output
Return exactly K distinct short option IDs from the supplied catalog. The executor validates them and normalizes them into label-only standard_candidates carrying canonical category IDs.
