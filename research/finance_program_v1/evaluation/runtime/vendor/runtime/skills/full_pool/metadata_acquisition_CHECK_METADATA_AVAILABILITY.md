---
name: CHECK_METADATA_AVAILABILITY
dsl: metadata_acquisition
required_inputs: [current_record]
optional_inputs: []
produces: [metadata_availability]
terminal: false
---
# Role
Report which approved metadata fields are present and which are absent.

# Evidence boundaries
Inspect only the supplied record. Never reconstruct hidden descriptions, values, labels, or ground truth.

# Output
Return JSON with `summary` and `data.metadata_availability`, containing `available` and `missing` lists.

# Failure
Return an explicit error when the record is unavailable or malformed.
