---
name: INFER_CATEGORY_FROM_TOPK
dsl: evidence_reasoning
required_inputs: [candidate_standard_categories, all_available_evidence]
optional_inputs: []
produces: [category, aligned_standard_entry]
terminal: false
---
# Role
Choose the single best-supported short option ID from the prior Top-K choices.

# Evidence boundary
The executor adds descriptions for only those K candidates. Do not use any option outside the supplied set, grading level, evaluation label, or long canonical category ID.

# Output
Return one supplied short option ID and a concise evidence summary. The executor maps it to the canonical category ID and binds the frozen Registry entry after the choice.
