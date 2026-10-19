---
name: GET_FIELD_METADATA
dsl: metadata_acquisition
required_inputs: [current_record]
optional_inputs: []
produces: [field_metadata]
terminal: false
---
# Role
Retrieve approved real metadata for this exact source record from the external
metadata store. This is a deterministic lookup, not a language-model guess.

# Evidence boundary
Only database name, table name, field description, and table description may be
returned. Match source ID, domain, and field name. Never expose the category,
grading level, split, or other target data. The Agent sees these fields only
after this Skill has been called and its Observation appended to state.

# Output
Return `field_metadata` with the available approved fields. Missing values
are omitted. The result is an Observation for subsequent Skill calls.
