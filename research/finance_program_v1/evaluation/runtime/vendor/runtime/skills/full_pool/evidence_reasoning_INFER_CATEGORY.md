---
name: INFER_CATEGORY
dsl: evidence_reasoning
required_inputs: [candidate_standard_categories, all_available_evidence]
optional_inputs: []
produces: [category, aligned_standard_entry]
terminal: false
---
# Role
Infers the final Category category from candidate standard categories and accumulated evidence.

# Provider and publication boundary
Use only actual resolved inputs and the supplied standard catalog. The executor supplies the single authoritative provider output_schema. Return only that provider JSON; do not return a data wrapper or canonical category/standard objects. The native header's produces list describes validated code publications, not the provider response shape. The executor maps accepted short IDs into the frozen catalog; a rank is not proof of semantic support. Unsupported classification must use the schema's explicit null outcome. Never invent metadata, candidate IDs, labels, grades or sufficient evidence.
