# DataClassifyGrading

A two-stage leaf-classification framework with strict JSON model outputs:

1. Stage 1 selects five prompt choice IDs from a runtime-loaded leaf registry.
2. Stage 2 selects one local bundle ID and immediately decodes it to the canonical `category_id`; formal joint runs also emit `data_level`.

## Repository data policy

The repository contains code, engineering configuration, synthetic examples/tests, and the approved `data/knowledge/**/*.json` knowledge base. Production datasets, registries, corpora, model inputs, reports, and training artifacts must remain local and untracked.

## Fresh-clone verification

Install the test extra so a fresh checkout has both `pytest` and `pyarrow`:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The examples in `cfg/task/*.example.json` are fabricated and are suitable for
smoke tests only. The VeRL server environment is separate; install the pinned
CUDA 12.8 stack from `requirements/verl.txt` on the authorized RTX PRO 6000
96GB host. The compatibility files intentionally require an exact runtime
vLLM lock rather than claiming an unverified compatibility version.

## Local classification

Supply every production asset by explicit path. Formal joint runs use a task
config whose metadata is `field_name` and a grading config whose ground-truth
field is `data_level`:

```bash
python -m script.verl.sft.evaluate_true_e2e \
  --model-path <local-model-dir> \
  --data <local-test.parquet> \
  --registry <local-registry.json> \
  --corpus <local-corpus.json> \
  --task-config <local-task.json> \
  --grading-config <local-grading.json> \
  --report <local-output-report.json>
```

`--report` is required and the output is runtime-local. Nothing under runtime
paths (datasets, standards, registries, corpora, parquet, models, reports, or
checkpoints) should be added to Git. Start with
[`docs/classification-runtime.md`](docs/classification-runtime.md) and follow
the exact server sequence in [`docs/server-runbook.md`](docs/server-runbook.md).
