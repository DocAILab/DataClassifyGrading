# Stage 2 hard-negative DPO

This experiment starts from the public `sft-method-refactor` SFT adapter and
adds a separate DPO LoRA. It does not modify or overwrite the SFT adapter.

The miner reads only `train.json`. It retrieves four semantic neighbors from
the label registry, scores the five answers with the frozen SFT policy, and
uses the correct answer as `chosen` and the highest-scoring wrong answer as
`rejected`. Only `field_name` is exposed as record metadata.

The paired evaluator reads only `val.json` and gives SFT and DPO exactly the
same five oracle candidates. Its result is a constrained Stage 2 comparison,
not a complete Stage 1-to-Stage 2 end-to-end score. The final report is
`val-evaluation/comparison_to_sft.json`.

The detached launcher requires `REGISTRY_PATH` and `TASK_CONFIG_PATH`:

```bash
REGISTRY_PATH=/absolute/leaf_registry.json \
TASK_CONFIG_PATH=/absolute/field_name_task.json \
bash src/method/dpo/script/start_stage2_hard_dpo.sh
```

Every phase is resumable. A one-update smoke must produce
`smoke-verification/SMOKE_VERIFIED` before full training begins. Storage checks
require 20 GiB free on `/root/autodl-tmp` and 35 GiB free in `/dev/shm`.
