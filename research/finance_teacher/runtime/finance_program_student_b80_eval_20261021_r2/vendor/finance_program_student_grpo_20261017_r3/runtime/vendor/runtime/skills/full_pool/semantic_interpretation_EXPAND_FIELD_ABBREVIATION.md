---
name: EXPAND_FIELD_ABBREVIATION
dsl: semantic_interpretation
required_inputs: [field_name]
optional_inputs: [domain]
produces: [field_abbreviation_semantics]
terminal: false
---
# Role (extractive-grounded-source-v1 revision)
Report source-grounded field meaning without speculative abbreviation expansion. This version explicitly replaces free natural-language semantics with a typed evidence object.

# Evidence boundaries
The executor supplies versioned grounding_context containing only initial field_name and already acquired/provenanced field_description, table_name and table_description. This is declared runtime context, not an extra Planner argument and not a hidden metadata lookup. No description is available until actual acquisition. Do not infer expansions, entities, types, values, labels, or consistency from an opaque name.

# Output
Return exactly summary and data.field_abbreviation_semantics. The latter is an object with exactly schema="grounded-source-v1", status="observed_description" or "unknown", field_name=exact context field_name, description=entire exact acquired field_description or null, source_consistency="not_adjudicated". summary equals status. With actual description, copy it and use observed_description; without description use unknown/null. No guesses, free-form claims, translations or conflict adjudication. Source consistency still requires independent review.

# Failure
Return an explicit error when the field name is empty.
