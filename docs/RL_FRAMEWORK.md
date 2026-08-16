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
this same interface; a future VeRL reward adapter should translate these facts
into algorithm-specific rewards rather than reimplement parsing.

### Training adapters

`agent.training.sft` owns messages-Parquet export, validation, the temporary
fixture candidate policy, and tokenizer-specific budget inspection.
`script/verl/sft/` contains the corresponding CLI and launcher adapters.
`script/verl/common/` contains environment and optional accelerator setup shared
by future VeRL algorithms.

The current fixture policy (ground truth followed by the first four registry
IDs) is deliberately SFT-local and is not a formal retrieval strategy.

## Dependency and configuration layout

`requirements/verl.txt` and `requirements/verl-cu128.constraints.txt` describe
the shared VeRL environment. There is no `base.txt` because the installable core
currently has no runtime dependency, and there is no `verl-sft.txt` because none
of the pinned packages is SFT-only. Add a narrower requirements layer only when
a real algorithm introduces a distinct dependency.

`cfg/task/` is independent of VeRL. Do not create `cfg/verl/common`, `sft`, or
`rl` placeholders; add an actual Hydra configuration there when a checked-in
launcher consumes it.

## When to add RL code

Do not create empty GRPO/PPO/DAPO packages. Add `agent.training.rl` and
`script/verl/rl/` only when the first real RL vertical slice has all of:

1. an agreed Stage 1 or Stage 2 reward contract and weights;
2. an explicit VeRL five-field Parquet adapter
   (`data_source`, `prompt`, `ability`, `reward_model`, `extra_info`);
3. a checked-in fixture that can exercise the reward adapter;
4. a CPU contract test and a small GPU smoke launcher.

At that point the VeRL-specific reward function should be a thin adapter around
`evaluate_stage1` or `evaluate_stage2`. Algorithm changes then belong in VeRL
configuration and algorithm-specific launchers, while task semantics remain in
one place.
