# Shougang Qwen3.5 Legacy Two-Stage SFT Design

## Objective

Fine-tune `Qwen/Qwen3.5-9B` with LoRA on the validated two-stage
`shougang-5415` training release, then evaluate the merged model on the exact
542-source test set and evaluator used for the completed non-training baseline.
This experiment must isolate the effect of SFT: the dataset, category universe,
metadata fields, prompts, decoding settings, test records, and metrics remain
unchanged.

## Fixed experiment contract

- Dataset: only the official `DocAILab/DCG` `shougang-5415` split.
- Canonical dataset identity: `shougang`.
- Official source split: 4,332 train, 541 validation, and 542 test records.
- Training release: 8,664 train rows and 1,082 validation rows, exactly one
  Stage-1 and one Stage-2 row per source record.
- Category universe: exactly 193 distinct `level_4` labels. Complete four-level
  paths remain provenance and derived corpus descriptions.
- Grading labels: literal `L1`, `L2`, `L3`, and `L4` from `data_level`. No
  unverified grading descriptions are introduced.
- Visible metadata: `field_name`, `table_name`, `field_description`, and
  `table_description`.
- Base model: the exact local `Qwen/Qwen3.5-9B` snapshot already used by the
  non-training baseline.
- No `sg_sft`, 233-class data, tool-trajectory parquet, generated thoughts,
  test rows, or validation rows may enter optimization.
- Formal evaluation uses greedy decoding, `seed=42`, thinking disabled, and
  the same registry, corpus, task config, grading config, and 542-row test
  release as the non-training baseline.

## Chosen architecture

Use the repository's legacy three-message Stage-1/Stage-2 SFT release directly
with the VeRL SFT trainer. Create a new persistent environment for VeRL 0.9.0
instead of mutating the inference environment. Apply and verify the repository's
required Qwen3.5 patches:

1. `chat_template-system-first.patch`;
2. `multiturn_sft_dataset-prefix-diff-answer-mask.patch`;
3. `losses-scheme-c.patch`.

The optional `agent_loop-debug.patch` is excluded. The launcher must fail
closed when the VeRL version is not 0.9.0 or required patch hashes do not match.
Training consumes the parquet `messages` column directly; no legacy mixture or
tool-trajectory adapter is inserted.

## Training configuration

- Precision: BF16.
- Adaptation: LoRA, rank 8, alpha 16, target modules `all-linear`.
- Base weights: frozen.
- Optimizer learning rate: `1e-4`.
- Maximum rendered sequence length: 4,096 tokens; truncation policy `error`.
- Gradient checkpointing: enabled for the formal run.
- Effective train batch size target: 32 examples. The launcher may realize
  this with the largest safe per-GPU micro-batch and gradient accumulation,
  but must record the realized values.
- Epochs: one formal epoch for the first controlled SFT baseline.
- Random seed: 42 wherever supported by the trainer.
- Hardware: one NVIDIA RTX PRO 6000 Blackwell 96 GB GPU.
- Checkpoints, environment records, logs, and status files live on persistent
  storage; model and hot parquet files may be staged on the fast disk.

The 8x answer-value weighting in the required VeRL patch applies only to
`answer` and `level` value spans in Stage 2. Stage-1 `candidates` output remains
at normal weight. This asymmetry is part of the recorded training condition and
must not be changed during the run.

## Implementation components

### Environment and patch gate

Create an idempotent setup path for a dedicated persistent virtual environment.
It records Python, CUDA, PyTorch, Transformers, VeRL, PEFT, and PyArrow versions;
applies the three required patches; and executes the tracked patch verifier.
Credentials are never stored in scripts, configuration, logs, or shell history.

### Formal launcher

Add a Qwen3.5-specific legacy SFT launcher that accepts explicit model, train,
validation, and output paths. It validates every required file and refuses a
non-empty output directory unless an explicit resume mode is selected. The
launcher records its full resolved configuration before invoking VeRL and uses
`nohup` only at the remote orchestration boundary, not internally.

### Training monitor

Run a persistent polling process beside the trainer. It records `running`,
`completed`, or `failed`; captures the producer PID, checkpoint inventory, log
tail, and completion time; and writes status atomically. Loss curves and the
final checkpoint are not declared valid solely because the producer exited
with code zero: the expected checkpoint metadata and adapter tensors must also
be present.

### Merge and evaluation

After training validation passes, merge the LoRA adapter into a standalone HF
model using the repository's merge utility. Run a two-source inference smoke
before the full test. Then evaluate all 542 official test sources with
`evaluate_true_e2e.py` and independently certify the final report using the
existing evaluation watcher.

## Gates and failure handling

1. **Static gate:** repository tests pass on Linux; SFT export and token-budget
   reports remain valid; train/validation/test hashes match the completed
   baseline release.
2. **Runtime gate:** VeRL is exactly 0.9.0, required patches verify, CUDA is
   available, and the model snapshot contains every indexed shard.
3. **Dataset seam gate:** patched VeRL can load a real Stage-1 row and a real
   Stage-2 row, render Qwen3.5 chat templates, produce non-empty assistant loss
   masks, and preserve the answer-value mask behavior.
4. **GPU smoke gate:** a two-step LoRA run produces finite loss, at least one
   valid adapter checkpoint, and no OOM or template exception.
5. **Formal run gate:** start the one-epoch run only after all earlier gates
   pass. If an OOM occurs, reduce only the micro-batch and increase gradient
   accumulation to preserve effective batch size 32; do not change the model,
   LoRA parameters, data, learning rate, or epoch count.
6. **Merge gate:** merged model loads from local files and completes strict
   Stage-1 and Stage-2 JSON inference on two fixed validation sources.
7. **Evaluation gate:** exactly 542 unique test sources are present and the
   independent watcher recomputes all critical counts.

Any failure stops the next stage and writes a structured status record. A
failed smoke run is never renamed or presented as a formal checkpoint.

## Success criteria and comparison

Operational success requires a verified LoRA checkpoint, a loadable merged HF
model, and a complete 542-source SFT evaluation report. The report is compared
against the frozen non-training baseline:

| Metric | Non-training baseline |
|---|---:|
| Stage-1 contract-valid rate | 96.13% |
| Stage-1 recall@5 | 47.42% |
| Category exact match | 34.32% |
| Data-level exact match | 25.09% |
| Strict joint exact match | 7.93% |

The first SFT run is scientifically valid even if one or more metrics fail to
improve, provided the data and execution contracts pass. Metric degradation is
reported rather than hidden by retuning on the test set. Hyperparameter changes
belong to a later experiment selected only from validation behavior.

## Deliverables

- Reproducible environment/patch verification record.
- Qwen3.5 SFT launcher and automated tests.
- Two-step smoke log and checkpoint audit.
- One-epoch training log, resolved configuration, status, and checkpoint.
- Merged standalone model and merge report.
- Two-source post-merge inference smoke report.
- Complete 542-source SFT evaluation and independent status report.
- A concise comparison against the frozen non-training baseline.

## Non-goals

This experiment does not perform full-parameter fine-tuning, QLoRA
quantization, tool-trajectory SFT, SFT-data generation from an external model,
RLOO/GRPO, test-set hyperparameter selection, category remapping, or mixing
with the 233-class `sg_sft` release.
