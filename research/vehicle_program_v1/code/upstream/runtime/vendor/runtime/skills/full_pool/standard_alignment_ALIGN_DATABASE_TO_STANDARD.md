---
name: ALIGN_DATABASE_TO_STANDARD
dsl: standard_alignment
required_inputs: [database_semantics, standard_candidates]
optional_inputs: []
produces: [align_database_to_standard]
terminal: false
---
# ALIGN_DATABASE_TO_STANDARD

Purpose: Aligns database-level semantics with candidate standard categories.

Observation: Explain how database context supports, contradicts or weakly matches each actual standard candidate; context remains auxiliary to target-field evidence.

Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.

Return exactly {"summary": "nonempty natural-language observation", "data": {...}}. Data keys must be exactly ["align_database_to_standard"]. Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.
