---
name: COMPARE_STANDARD_CANDIDATES
dsl: standard_alignment
required_inputs: [standard_candidates, semantic_evidence]
optional_inputs: []
produces: [standard_comparison, aligned_standard_entry]
terminal: false
---
# Role
Compare retrieved candidates against the current semantic evidence and identify the best-supported entry.

# Evidence boundaries
Compare only supplied candidates. Do not add categories, hidden metadata, labels, or ground truth.

# Output
Return exactly one JSON object with exactly `decision`, `option_id`, and `reason`. For a supported candidate use `{"decision":"match","option_id":"C001","reason":"brief"}`, choosing an exact short `option_id` from the actual supplied `standard_candidate_details`. When no supplied candidate is supported use `{"decision":"no_match","option_id":null,"reason":"brief"}`. `reason` must be a nonempty brief string of at most 240 characters. Do not emit Markdown, duplicate fields, a summary/data wrapper, canonical category IDs, or full standard objects. The executor binds a successful short ID to the frozen standard and publishes the internal `standard_comparison` and `aligned_standard_entry`.

# Failure
Absent inputs fail validation before generation. `no_match` is a model judgment and fails without publishing alignment or classification. Invalid format, actual output truncation, and backend generation errors are reported separately; do not invent a match.
