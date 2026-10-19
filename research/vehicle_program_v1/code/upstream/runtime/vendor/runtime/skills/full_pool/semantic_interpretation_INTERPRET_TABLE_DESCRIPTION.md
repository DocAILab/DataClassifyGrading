---
name: INTERPRET_TABLE_DESCRIPTION
dsl: semantic_interpretation
required_inputs: [table_description]
optional_inputs: [table_name, domain]
produces: [table_description_semantics]
terminal: false
---
# Role
Explain the business context of an observed table description, without assigning a category.

# Evidence boundary
Use only the supplied real description and optional supplied context. The table's business
topic is context for the target field, not proof that all fields have one category.
Never invent values, labels or grading levels.

# Output
Return JSON with a short `summary` and `data.table_description_semantics` as concise text.

# Failure
Do not infer from an absent or empty table description.
