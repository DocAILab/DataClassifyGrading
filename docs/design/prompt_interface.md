# Prompt Interface (choice protocol)

Status: **frozen** (Phase 6 / PR #8, merged on master). The model never sees or
generates canonical `category_id`; it speaks a compact **choice protocol** that
is decoded to canonical ids at the LLM boundary.

## 1. Why a choice protocol

- Canonical ids are internal identity (`finance:业务.账户信息.基本信息`) — long,
  verbose, and **must not become model actions** (generated ids would be
  brittle and leak internal taxonomy structure).
- Choice ids are compact (`"1".."N"`), deterministic, and provide a **compact
  action representation** (they do not reduce the number of categories). This
  is a **decoupling/identity** decision, not only a token reduction one (see
  `docs/prompt_length_analysis.md` §6).

## 2. Mapping (`src/agent/task/prompt_choices.py`)

`PromptChoiceRegistry` maps canonical `category_id` ↔ global choice id
(`"1".."N"` following `LeafRegistry.categories` order, stable across runs)
plus a **display name** = shortest unique path suffix (leaf name when unique,
parent-qualified otherwise). Choice ids exist only inside prompts/model outputs;
they are decoded back immediately and never written to canonical records.

## 3. Stage 1 (retrieve 5 candidates over the full registry)

- Prompt: catalog of `[choice_id, display_name]` pairs for the whole registry.
- Model output: `{"candidates": ["3", "7", "12", "19", "24"]}` (global choice ids).
- Checks (shared, never raise on model output): JSON/schema → exactly 5 →
  unique → choice_id in catalog → decode to canonical `category_id` tuple.

## 4. Stage 2 (pick one answer from the 5-candidate bundle)

- Prompt: candidate bundle with **local** ids `"1".."5"` in candidate order +
  display name + corpus description/examples.
- Model output: `{"answer": "2"}` (local bundle id).
- Checks (shared): JSON/schema → answer ∈ `"1".."5"` (strict, no coercion) →
  positional decode against the canonical candidates.

## 5. Shared decode layer (Phase 7 / PR #9)

`agent.task.parser.check_stage1_choices` / `check_stage2_choices` →
`ChoiceParseResult` (format/constraint validity + decoded canonical ids +
`canonical_view()`). **Both evaluation and reward consume this single
implementation** — no second copy of choice validation:

- `agent.evaluation.evaluate_stage1_choices` / `evaluate_stage2_choices`
- `agent.training.rl.reward_stage1_choices` / `reward_stage2_choices` /
  `reward_for_choice_result` (decode first, then the unchanged reward table)

Stage-2 answer — `docs/design/data_contract.md` §6 — ground truth is NOT
required to be among the candidates (a Stage-1 recall miss is a legal state).

## 6. Files

- `src/agent/task/prompt_choices.py` — registry + encode/decode primitives
- `src/agent/task/parser.py` — shared never-raise choice-aware checks
- `src/agent/task/prompts.py` — prompt builders (`build_stage1_prompt`,
  `build_stage2_prompt`, `stage1_answer`, `stage2_answer`)
- Exports/layout facts: `docs/SFT_BASELINE.md`, `docs/RL_FRAMEWORK.md`;
  historical design: `docs/phase_reports/stage2_contract_design.md`
