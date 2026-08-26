# DataClassifyGrading

A leaf-classification framework with strict JSON model outputs. The formal
shougang RLOO path uses a native Qwen3.5 tool trajectory rather than a
synthetic two-stage user bridge:

1. The model may answer directly or call deterministic `search_categories`.
2. It may optionally inspect candidates with `get_category_details` or
   `get_category_examples`; tool observations are excluded from policy loss.
3. The terminal assistant response is exactly `{"answer":"<choice_id>",
   "level":"<data_level>"}` and receives one strict joint exact-match reward.

The tools read only the runtime registry/corpus JSON. They use no embedding
model, vector store, network retrieval, or per-sample ground truth.

## Repository data policy

The repository contains code, engineering configuration, synthetic examples/tests, and the approved `data/knowledge/**/*.json` knowledge base. Production datasets, registries, corpora, model inputs, reports, and training artifacts must remain local and untracked.

## Fresh-clone verification

Install the test extra so a fresh checkout has both `pytest` and `pyarrow`:

```bash
python -m pip install -e ".[test]"
python -m pytest
```

The examples in `cfg/task/*.example.json` are fabricated and are suitable for
smoke tests only. The VeRL server environment is separate; the native-tool
RLOO launcher requires the validated core in
`requirements/verl-qwen35-cu130.txt` (VeRL 0.9.0, vLLM 0.27.1,
Transformers 5.10.4, PyTorch 2.13.0+cu130) on the authorized RTX PRO 6000
96GB host. Do not mix it with the archived VeRL 0.8 environment.

## Local classification

Supply every production asset by explicit path. Formal shougang runs use a task
config whose metadata is `field_name` and `table_name`, plus a grading config
whose ground-truth field is `data_level`:

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
checkpoints) should be added to Git. The reference example uses
`/absolute/path/to/Qwen3.5-9B`; old Qwen2.5 token limits are invalid, so
re-measure the selected model's chat-template budget with the repository token
budget CLI before training. Start with
[`docs/classification-runtime.md`](docs/classification-runtime.md) and follow
the exact server sequence in [`docs/server-runbook.md`](docs/server-runbook.md).
