# Reusable VeRL post-training framework

This repository treats VeRL as an external, pinned training engine. The
repository owns the classification task, evaluation rules, dataset adapters,
launchers, and tests; it does not vendor or fork VeRL.

## Responsibility boundary

The preprocessing and corpus pipelines are maintained separately under
`script/preprocessing/` and `script/corpus/`. The training framework does not
import their implementation. Its input seam is a directory containing the
normalized `train.json`, `val.json`, and `test.json` splits documented in
`data/README.md` and `docs/新数据集运行说明.md`.

Label cleaning, formal taxonomy mapping, corpus construction, and hard-negative
selection remain preprocessing/task decisions. Training adapters must not
silently add such rules.

## Modules and seams

### Task module

`agent.task` owns algorithm-independent behavior:

- explicit leaf registry and prompt-visible metadata contracts;
- deterministic Stage 1 and Stage 2 prompts;
- strict JSON output parsing without ChatML serialization.

Its parser accepts only these output shapes:

```json
{"candidates":["id1","id2","id3","id4","id5"]}
{"answer":"id1"}
```

### Evaluation module

`agent.evaluation` is pure Python and has no VeRL dependency:

```python
from agent.evaluation import evaluate_stage1, evaluate_stage2

stage1 = evaluate_stage1(solution, ground_truth=gold, registry=registry)
stage2 = evaluate_stage2(
    solution,
    ground_truth=gold,
    candidates=candidates,
    registry=registry,
)
```

The results keep format validity, task-contract validity, and task outcome
separate. Stage 2 can still be evaluated when Stage 1 omitted the ground truth;
it reports an incorrect outcome without deciding whether training should stop,
mask, or reject. The results intentionally do not assign reward weights. SFT validation uses
this same interface; the RL reward adapter translates these facts into
stage-specific task rewards rather than reimplementing parsing.

Stage 4A added the unified parser layer in `agent.task.parser`
(`check_stage1_output` / `check_stage2_output` -> `ParseResult`) that never
raises on model output and separates format validity from constraint validity;
the evaluator and the reward adapter both consume it so there is exactly one
contract implementation.

Phase 6 introduced the prompt-facing choice protocol
(PromptChoiceRegistry, global Stage 1 ids and local Stage 2 ids), keeping
the canonical category_id out of prompts and model outputs.

Phase 7 adds the shared choice-aware parser/decode layer in the same parser
module (`check_stage1_choices` / `check_stage2_choices` ->
`ChoiceParseResult`): model rollouts answer with choice ids (Stage 1 global
ids, Stage 2 local bundle ids), which this layer validates (JSON/schema,
count, uniqueness, known ids / 1..5) and decodes to canonical category_ids
BEFORE the existing evaluation (`evaluate_stage1_choices` /
`evaluate_stage2_choices`) and reward entry points
(`reward_stage1_choices` / `reward_stage2_choices`) apply, completing the
task-level choice-to-canonical reward contract. Choice validation is
implemented exactly once; `RewardConfig`, reward numbers, canonical
registry/ground truth/candidates are unchanged. `RewardResult` gained a
`constraint_valid` flag (additive) so "valid but wrong" (partial credit) is
distinguishable from "invalid" (zero).

### Training adapters

`agent.training.sft` owns messages-Parquet export, validation, the temporary
fixture candidate policy, and tokenizer-specific budget inspection.
`agent.training.rl` owns the unified RL data/parser/reward contracts:

- `sample.py` -- resolved-only RL samples (RlSample: stage/source_id/messages/
  ground_truth/candidates/metadata/reward metadata); ground truth is
  exclusively ``target.category_id``;
- `reward.py` -- task reward (RewardConfig / RewardResult, reward_stage1 /
  reward_stage2, plus the choice-protocol entry points
  reward_stage1_choices / reward_stage2_choices / reward_for_choice_result
  that decode choice ids to canonical category ids via the shared choice
  parser BEFORE the same reward table applies); weights: 0.0 invalid, 0.3 valid-but-miss (Stage 1),
  1.0 correct, configurable ``stage2_partial`` for Stage 2
  (provisional / pending task-policy confirmation, default 0.5 is NOT a
  finalized reward policy);
- `dataset.py` -- VeRL v0.8.0 five-field RL parquet exporter/validator
  (`data_source` / `prompt` / `ability` / `reward_model` / `extra_info`; the
  prompt is system+user only, no assistant gold).

`script/verl/rl/` contains the corresponding CLI adapters (export/validate).
`script/verl/common/` contains environment and optional accelerator setup shared
by VeRL algorithms.

The current fixture policy (ground truth followed by the first four registry
IDs) is deliberately shared by SFT and RL as a baseline and is NOT the
production Stage 1 retrieval strategy.

## Dependency and configuration layout

`requirements/verl.txt` and `requirements/verl-cu128.constraints.txt` describe
the shared VeRL environment. There is no `base.txt` because the installable core
currently has no runtime dependency, and there is no `verl-sft.txt` because none
of the pinned packages is SFT-only. Add a narrower requirements layer only when
a real algorithm introduces a distinct dependency.

`cfg/task/` is independent of VeRL. Do not create `cfg/verl/common`, `sft`, or
`rl` placeholders; add an actual Hydra configuration there when a checked-in
launcher consumes it.

## RL landing status (after Stage 4A)

Stage 4A established the reusable RL data / parser / reward contract:

- agreed reward table (0.0 / 0.3 / 1.0 and a configurable Stage 2 partial);
- explicit VeRL five-field Parquet adapter
  (`data_source`, `prompt`, `ability`, `reward_model`, `extra_info`),
  verified against verl v0.8.0 `RLHFDataset` + reward manager
  (`non_tensor_batch["reward_model"]["ground_truth"]`);
- CPU contract tests (`tests/rl/`) and real-data e2e on all four datasets.

Still landing with the first real RL vertical slice (M4):

1. a checked-in fixture that runs the reward adapter through a real VeRL
   reward loop;
2. a small GPU smoke launcher.

The task-level choice-to-canonical contract is now complete on master: Phase
6 (prompt-facing choice protocol, PromptChoiceRegistry with global Stage 1
ids and local Stage 2 ids) and Phase 7 (shared choice-aware parser/decode
layer consumed by evaluation and reward; see the Evaluation module) are both
merged.

## When to add training-algorithm RL code

Do not create empty GRPO/PPO/DAPO packages or `cfg/verl/rl` placeholders.
The training algorithm, rollout engine, `script/verl/rl/run.sh` launcher and
Hydra configuration are added only when the first real RL vertical slice ships:

1. agreed task reward weights (Stage 4A defaults + confirmed Stage 2 partial);
2. a checked-in fixture that exercises the reward adapter through VeRL;
3. a CPU contract test and a small GPU smoke launcher.

At that point the VeRL-specific reward function is a thin adapter around
`reward_stage1_choices` / `reward_stage2_choices` (Phase 12, `verl_adapter.py` -- routes
VeRL `compute_score` by `<dataset>/stage<1|2>` to the choice-aware reward,
which parses choice ids via the shared choice parser and decodes to canonical
category ids before the unchanged reward table); algorithm changes then
belong in VeRL configuration and algorithm-specific launchers, while task
semantics remain in one place.
