---
name: INTERPRET_FIELD_DESCRIPTION
dsl: semantic_interpretation
required_inputs: [field_description]
optional_inputs: [field_name, domain]
produces: [field_description_semantics]
terminal: false
---
# Role (extractive-grounded-source-v1 revision)
Report the observed description with exact source support rather than generating unverified interpretations or claiming that an opaque name contradicts it.

# Evidence boundary
The executor supplies versioned grounding_context containing initial field_name and already acquired/provenanced field_description, table_name and table_description. This declared runtime context is not a Planner argument or hidden lookup. The required field_description must be the real acquired value. A source description is an observation, not proof of semantic truth or agreement with the name. Do not guess values, types, labels or conflict.

# Output
Return exactly summary and data.field_description_semantics. The latter is an object with exactly schema="grounded-source-v1", status="observed_description", field_name=exact context field_name, description=entire exact acquired field_description, source_consistency="not_adjudicated". summary equals status. No extra fields, interpretations, translation or conflict adjudication. If the required description is absent, original input validation fails rather than fabricating a source.

# Failure
Do not infer from an absent or empty field description.
