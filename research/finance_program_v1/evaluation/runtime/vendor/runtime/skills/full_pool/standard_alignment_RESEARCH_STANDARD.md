---
name: RESEARCH_STANDARD
dsl: standard_alignment
required_inputs: [refined_semantic_query]
optional_inputs: []
produces: [standard_candidates]
terminal: false
---
# Role
Performs a subsequent standard search after semantic refinement.

# Provider and publication boundary
Use only actual resolved inputs and the supplied standard catalog. The executor supplies the single authoritative provider output_schema. Return only that provider JSON; do not return a data wrapper or canonical category/standard objects. The native header's produces list describes validated code publications, not the provider response shape. The executor maps accepted short IDs into the frozen catalog; a rank is not proof of semantic support. Unsupported classification must use the schema's explicit null outcome. Never invent metadata, candidate IDs, labels, grades or sufficient evidence.
