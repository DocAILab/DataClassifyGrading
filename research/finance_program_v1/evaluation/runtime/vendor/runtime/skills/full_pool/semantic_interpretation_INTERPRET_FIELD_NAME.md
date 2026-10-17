---
name: INTERPRET_FIELD_NAME
dsl: semantic_interpretation
required_inputs: [field_name]
optional_inputs: [domain, field_abbreviation_semantics]
produces: [field_semantics]
terminal: false
---
# Role (extractive-grounded-source-v1 revision)
Report source-grounded field meaning without speculative name interpretation. This revision replaces free natural-language semantics with a typed evidence object; it does not claim a source description proves name-description consistency.

# Evidence boundaries
The executor supplies versioned grounding_context containing initial field_name and only already acquired/provenanced field_description, table_name and table_description. This declared runtime context is not an extra Planner argument or hidden lookup. No description is available before actual acquisition. An upstream model expansion cannot establish a verified business meaning. Do not infer missing meanings, values, types or labels.

# Output
Return exactly summary and data.field_semantics. The latter is an object with exactly schema="grounded-source-v1", status="observed_description" or "unknown", field_name=exact context field_name, description=entire exact acquired field_description or null, source_consistency="not_adjudicated". summary equals status. With a real acquired description copy it/use observed_description; otherwise unknown/null. No guesses, free prose, translations, abbreviations or adjudicated conflict/consistency claims. Actual source consistency requires independent review.

# Failure
Return an explicit error when the field name is empty.
