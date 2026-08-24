# Stage 2 Hard-Negative DPO Design

## Objective

Measure whether offline preference optimization can improve the Shougang
`field_name -> classification.level_4` decision after the repository's public
SFT baseline. The first experiment isolates Stage 2 candidate selection so a
failure in Stage 1 retrieval cannot hide the DPO effect.

## Immutable baseline

- Source revision: `origin/sft-method-refactor` at
  `7672ddc13cef8d9e4ac3b30f9d96f485c4276609`.
- Base model: Qwen2.5-7B-Instruct.
- Policy initialization: the LoRA produced by that public SFT pipeline.
- Reference policy: the same SFT state, frozen for the whole DPO run.
- DPO must write a new adapter and must never overwrite the SFT adapter.

The implementation lives in a separate `src/method/dpo` package. Existing SFT
behavior is not modified.

## Data boundary and preference construction

Only `train.json` may be used to construct preferences. `val.json` is reserved
for evaluation and `test.json` is forbidden: the pipeline must reject `test`
before resolving or opening it and must report `real_test_split_read=false`.

For every labeled training record:

1. Expose only `metadata.field_name` as record metadata.
2. Construct five candidates containing the golden level-4 label and four
   deterministic difficult negatives. Difficult negatives come from the SFT
   model's highest-scoring wrong labels; random negatives are not used as DPO
   rejections.
3. Shuffle the five candidates deterministically to remove position leakage.
4. Build the same professional English Stage 2 prompt used by the task
   contract, including candidate descriptions.
5. Store the correct JSON answer as `chosen` and the highest-scoring wrong
   candidate as `rejected`.

Each row records `source_id`, candidates, scores/provenance, golden position,
chosen/rejected labels, seed, model identity, and prompt checksum. The exporter
rejects duplicate candidates, OOV labels, chosen/rejected equality, a missing
golden label, split overlap, or leaked metadata.

## Training

The policy and reference start from the same SFT weights. The reference is
frozen; only a fresh policy LoRA is trainable. Initial configuration:

- LoRA rank 16, alpha 32;
- BF16 and gradient checkpointing;
- micro-batch 1, gradient accumulation 8;
- learning rate `5e-7`;
- DPO beta `0.1`;
- one epoch, seed 42, checkpoint every 100 optimizer steps.

Before the full run, a smoke test must complete a real forward pass, backward
pass, optimizer update, metric write, and checkpoint/adapter save. It must also
prove that policy parameters changed while reference parameters did not.

The full run is recoverable from its latest verified checkpoint and writes
configuration, data audit, trainer metrics, storage/GPU measurements, adapter
checksum, completion/failure markers, and a training report.

## Evaluation and decision rule

Evaluate SFT and DPO on the same 2,028 unique validation `source_id` values and
the same deterministic hard-candidate sets. Report:

- Stage 2 accuracy and Macro-F1;
- JSON compliance, OOV and invalid counts;
- per-class and frequency-bucket metrics;
- paired SFT-to-DPO wins, losses and unchanged samples;
- McNemar significance;
- model/adapter checksums, time and peak GPU memory.

The experiment is considered promising only if DPO exceeds the SFT Stage 2
accuracy, does not reduce Macro-F1, keeps format compliance near 100%, and the
gain is reproducible rather than a sub-one-percentage-point single-seed change.
If the first seed is promising, seed 137 is run as confirmation. If Stage 2
does not improve, the pipeline stops without extending DPO to Stage 1.

## Storage safety

Before training, inventory the persistent disk and `/dev/shm`. Deletion is
limited to verified completed/abandoned experiments' transient checkpoints,
failed fragments, stale caches, and other reproducible files. Preserve all
datasets, base models, permanent LoRA adapters, reports, logs, checksums, and
partial predictions. Record every deleted path and reclaimed byte count in a
cleanup report. Require sufficient free space before smoke and full training.

## Error handling and tests

Implementation follows test-first development. Tests cover train-only access,
metadata leakage, deterministic hard-negative selection and shuffling,
preference validity, reference freezing, parameter updates, checkpoint resume,
storage guards, and paired evaluation. On remote failure, retain completed
markers and artifacts, diagnose from logs, add a regression test, and resume
only the unfinished stage.
