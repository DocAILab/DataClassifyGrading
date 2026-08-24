# Stage 2 Hard-Negative DPO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a recoverable Stage 2 hard-negative DPO experiment on top of the public GitHub SFT adapter and compare it fairly with the unchanged SFT policy.

**Architecture:** New code lives under `src/method/dpo`; it reuses task prompts and evaluation primitives but does not modify SFT behavior. A train-only miner creates deterministic hard preference pairs, TRL trains a fresh LoRA against a frozen SFT reference, and a paired val evaluator scores the same candidate sets with both policies.

**Tech Stack:** Python 3.11, PyTorch 2.8, Transformers 5.15, PEFT 0.20, TRL 0.24, Datasets 5, PyArrow 25, pytest 9, Qwen2.5-7B-Instruct.

## Global Constraints

- Source baseline is `origin/sft-method-refactor` commit `7672ddc13cef8d9e4ac3b30f9d96f485c4276609`.
- Record input is only `metadata.field_name`; supervision is `classification.level_4`.
- Preference construction reads train only; val is evaluation only; test is rejected before path access.
- The public SFT adapter is immutable and DPO saves a separate LoRA.
- Initial DPO settings: rank 16, alpha 32, BF16, gradient checkpointing, micro-batch 1, accumulation 8, learning rate `5e-7`, beta `0.1`, one epoch, seed 42, save every 100 steps.
- Do not push GitHub during this experiment.

---

### Task 1: Preference row contract and train-only exporter

**Files:**
- Create: `src/method/dpo/__init__.py`
- Create: `src/method/dpo/preference_data.py`
- Create: `tests/method/dpo/test_preference_data.py`

**Interfaces:**
- Consumes: `LeafRegistry`, `TaskConfig`, `build_stage2_prompt`, JSONL label-score records.
- Produces: `select_hard_candidates(...) -> list[str]`, `build_preference_row(...) -> dict`, `export_preferences(...) -> dict`.

- [ ] Write failing tests proving deterministic golden-plus-four-hard candidates, shuffled golden positions, exact chosen/rejected JSON, OOV/duplicate rejection, `field_name`-only prompts, and rejection of any split other than `train` before file access.
- [ ] Run `python -m pytest tests/method/dpo/test_preference_data.py -q`; expect import failure because `method.dpo` does not exist.
- [ ] Implement immutable score parsing, deterministic SHA-256-seeded shuffling, prompt construction, preference JSONL/Parquet output, and an audit report containing `requested_splits=["train"]` and `real_test_split_read=false`.
- [ ] Re-run the focused test and `python -m pytest -q`; expect all baseline and DPO tests to pass.
- [ ] Commit with `git commit -m "feat: export train-only hard DPO preferences"`.

Required row shape:

```python
{
    "source_id": source_id,
    "prompt": prompt,
    "chosen": json.dumps({"answer": golden}, ensure_ascii=False),
    "rejected": json.dumps({"answer": hardest_wrong}, ensure_ascii=False),
    "candidates": shuffled_candidates,
    "ground_truth": golden,
    "rejected_label": hardest_wrong,
    "hard_negative_scores": score_map,
    "metadata": {"field_name": field_name},
    "seed": seed,
}
```

### Task 2: Memory-bounded SFT hard-negative miner

**Files:**
- Create: `src/method/dpo/label_scoring.py`
- Create: `src/method/dpo/script/mine_preferences.py`
- Create: `tests/method/dpo/test_label_scoring.py`

**Interfaces:**
- Consumes: base model path, immutable SFT LoRA path, registry, train records.
- Produces: resumable `label_scores.jsonl` with one complete row per `source_id` and a checksum-bearing mining report.

- [ ] Write failing unit tests for completion-only mean log-prob, stable tie-breaking by registry order, batching equivalence, completed-`source_id` resume, and refusal to load val/test.
- [ ] Run the focused tests; expect missing scorer symbols.
- [ ] Implement label-token-only logits slicing so the code never materializes full-sequence log-softmax, batch candidate labels, append atomically per source, and record peak GPU memory and elapsed time.
- [ ] Add CLI arguments `--input-dir`, `--registry`, `--task-config`, `--model`, `--sft-adapter`, `--output-dir`, `--batch-size`, `--seed`; do not expose a split option and always read `train.json`.
- [ ] Run focused and full tests, then commit `feat: mine SFT hard negatives for DPO`.

### Task 3: DPO policy/reference loading and trainer

**Files:**
- Create: `src/method/dpo/training.py`
- Create: `src/method/dpo/script/train.py`
- Create: `tests/method/dpo/test_training.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: preference Parquet, Qwen base model, public SFT adapter.
- Produces: fresh DPO LoRA, `trainer_metrics.jsonl`, checkpoint directories, `training_report.json`.

- [ ] Write failing tests for explicit TRL/PEFT version checks, frozen reference parameters, fresh trainable policy adapter, policy/reference initial-logit equality, non-overwriting output paths, and run-config serialization.
- [ ] Run focused tests and verify the desired missing-function failures.
- [ ] Implement in-memory SFT merge for two independent model instances, freeze reference, add rank-16/alpha-32 LoRA to policy, and configure `DPOTrainer` with beta 0.1, learning rate `5e-7`, batch 1, accumulation 8, BF16, checkpointing, one epoch and save-steps 100.
- [ ] Save only the DPO adapter plus exact base/SFT checksums. Refuse to start if output resolves inside the SFT adapter directory.
- [ ] Run focused and full tests, then commit `feat: train DPO from immutable SFT baseline`.

The trainer's immutable configuration projection is:

```python
{
    "beta": 0.1,
    "learning_rate": 5e-7,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "num_train_epochs": 1.0,
    "bf16": True,
    "gradient_checkpointing": True,
    "save_steps": 100,
    "seed": 42,
}
```

### Task 4: Real-update smoke verification

**Files:**
- Create: `src/method/dpo/smoke_verification.py`
- Create: `tests/method/dpo/test_smoke_verification.py`

**Interfaces:**
- Consumes: before/after policy tensors, before/after reference tensors, trainer state, adapter/checkpoint paths.
- Produces: `smoke_report.json` and `SMOKE_VERIFIED` only when every invariant passes.

- [ ] Write failing tests requiring finite loss, nonzero finite grad norm, positive policy update norm, zero reference update norm, one optimizer step, metrics, adapter files and checkpoint files.
- [ ] Run the test and observe missing verifier failure.
- [ ] Implement tensor checksum/update statistics and a fail-closed verifier; never create `SMOKE_VERIFIED` on partial success.
- [ ] Run focused/full tests and commit `test: verify real DPO optimizer updates`.

### Task 5: Paired SFT-versus-DPO Stage 2 evaluator

**Files:**
- Create: `src/method/dpo/evaluation.py`
- Create: `src/method/dpo/script/evaluate.py`
- Create: `tests/method/dpo/test_evaluation.py`

**Interfaces:**
- Consumes: val records, deterministic oracle hard-candidate rows, SFT policy, SFT+DPO policy.
- Produces: two prediction JSONL files and `comparison_to_sft.json`.

- [ ] Write failing tests for 2,028-style unique pairing, accuracy, Macro-F1, JSON compliance, OOV/invalid counts, win/loss/tie counts, exact McNemar p-value, per-class metrics, and rejection of test.
- [ ] Run focused tests and observe missing evaluator failure.
- [ ] Implement resumable same-candidate inference and paired summaries. Enforce identical `source_id`, prompts, candidates and ground truth across both policies.
- [ ] Run focused/full tests and commit `feat: compare DPO and SFT on paired val rows`.

### Task 6: Storage-safe recoverable remote pipeline

**Files:**
- Create: `src/method/dpo/storage.py`
- Create: `src/method/dpo/script/run_stage2_hard_dpo.sh`
- Create: `src/method/dpo/script/start_stage2_hard_dpo.sh`
- Create: `tests/method/dpo/test_pipeline.py`
- Create: `docs/DPO_STAGE2.md`

**Interfaces:**
- Consumes: persistent workspace, existing model/SFT adapter/data paths.
- Produces: phase markers, `status.json`, `pipeline.log`, `cleanup_report.json`, final comparison and `COMPLETE`/`FAILED`.

- [ ] Write failing tests for minimum 20 GiB persistent free space, minimum 35 GiB `/dev/shm`, preservation of datasets/models/adapters/reports, phase resume, detached `setsid` launch, and test-path prohibition.
- [ ] Run focused tests and observe missing storage/pipeline behavior.
- [ ] Implement a dry-run inventory and explicit allow-list cleanup recorder; the launcher itself never recursively deletes a computed broad path.
- [ ] Implement phases: audit, mine, export, smoke, full, val SFT, val DPO, compare. Each phase writes an atomic marker and skips only after artifact validation.
- [ ] Document exact commands, artifact paths, recovery behavior and metrics.
- [ ] Run focused/full tests and commit `feat: add recoverable Stage2 DPO pipeline`.

### Task 7: Remote verification and experiment

**Files:**
- Remote worktree: `/root/autodl-tmp/worktrees/DataClassifyGrading-stage2-hard-dpo`
- Remote artifacts: `/root/autodl-tmp/artifacts/shougang/stage2-hard-dpo-v1`

**Interfaces:**
- Consumes: implementation bundle, existing Qwen model, public random-shuffled SFT LoRA, Shougang train/val.
- Produces: verified smoke, full DPO adapter, paired val comparison.

- [ ] Bundle the local branch without pushing GitHub; transfer it and create the isolated remote worktree at the recorded commit.
- [ ] Verify remote `pip check`, library versions, GPU BF16, base model, SFT adapter checksum, dataset paths, 27+ GiB free persistent space and 35+ GiB `/dev/shm`.
- [ ] Run the full pytest suite remotely; expect zero failures.
- [ ] Launch mining and preference export; validate train-only audit, candidate uniqueness, OOV=0 and prompt metadata fields exactly `[field_name]`.
- [ ] Run one-update smoke and require `SMOKE_VERIFIED` before full training.
- [ ] Launch full training detached from SSH and resume from the latest valid checkpoint on interruption.
- [ ] Evaluate SFT and DPO on the same 2,028 val IDs, verify `real_test_split_read=false`, and report accuracy, Macro-F1, paired changes, McNemar, compliance, OOV/invalid, time, memory and checksums.
- [ ] If seed 42 meets the design decision rule, run seed 137; otherwise stop without Stage 1 DPO.
