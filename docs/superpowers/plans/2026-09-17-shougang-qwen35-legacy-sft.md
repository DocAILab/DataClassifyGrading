# Shougang Qwen3.5 Legacy Two-Stage SFT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train and evaluate a reproducible Qwen3.5-9B LoRA baseline on the validated shougang-5415 Stage-1/Stage-2 SFT release.

**Architecture:** A Python experiment CLI validates immutable runtime inputs, the VeRL 0.9/Qwen3.5 patch gate, and the fixed training contract before constructing the existing `run.sh` command. A separate polling CLI monitors the background producer and atomically certifies either a usable checkpoint or a structured failure; remote orchestration performs a two-step smoke before the one-epoch formal run.

**Tech Stack:** Python 3.12, pytest, PyTorch 2.8 BF16, Transformers 5.10.4, VeRL 0.9.0, PEFT/LoRA, PyArrow, one RTX PRO 6000 Blackwell 96 GB GPU.

## Global Constraints

- Use only the official shougang-5415 split: 4,332 train, 541 validation, 542 test sources and exactly 193 level-4 classes.
- Optimize only the validated 8,664-row train parquet; validation and test rows never enter optimization.
- Use the exact Qwen3.5-9B snapshot and runtime assets from the frozen non-training baseline.
- Use BF16, LoRA rank 8, alpha 16, `all-linear`, learning rate `1e-4`, maximum length 4,096, effective batch size 32, and one formal epoch.
- Apply the three required VeRL 0.9 Qwen3.5 patches and exclude the optional debug patch.
- Preserve seed 42 and the same 542-source greedy, thinking-disabled evaluator used by the non-training baseline.
- Never persist SSH passwords, Hugging Face tokens, or ModelScope tokens.

---

### Task 1: Fail-closed Qwen3.5 SFT experiment CLI

**Files:**
- Create: `script/verl/sft/qwen35_legacy_experiment.py`
- Create: `tests/sft/test_qwen35_legacy_experiment.py`
- Modify: `script/verl/sft/run.sh`

**Interfaces:**
- Consumes: explicit model directory, train/validation parquet, output directory, Python executable, repo root, smoke/formal mode, and optional resume checkpoint.
- Produces: `ExperimentConfig`, `validate_runtime(config)`, `build_command(config)`, a deterministic resolved-config JSON, and an optional launched VeRL process.

- [ ] **Step 1: Write failing configuration and command tests**

  Test that formal mode emits the fixed BF16/LoRA/learning-rate/length/epoch overrides, smoke mode emits exactly two steps, all paths are separate argv elements, and no credential-like environment value appears in the resolved config.

- [ ] **Step 2: Run tests and confirm the module is missing**

  Run: `python -m pytest tests/sft/test_qwen35_legacy_experiment.py -q`

  Expected: collection failure for `script.verl.sft.qwen35_legacy_experiment`.

- [ ] **Step 3: Implement the minimal configuration and command builder**

  Use an immutable dataclass, fixed defaults from the design, explicit path validation, output-directory refusal, and an argv list ending in the repository's `script/verl/sft/run.sh` Hydra overrides. Add a `--dry-run` mode that writes and prints the resolved configuration without launching training.

- [ ] **Step 4: Add runtime/version/patch rejection tests**

  Cover missing model shards, missing parquet, wrong VeRL version, failed patch verifier, a non-empty new output directory, and resume mode without an existing checkpoint.

- [ ] **Step 5: Implement runtime and patch gates**

  Verify indexed model shards, parquet schemas and row counts, CUDA availability, VeRL 0.9.0, and the tracked `verify_verl_patches.sh` result before launch. Update `run.sh`'s stale 0.8 label without changing its argv-preserving behavior.

- [ ] **Step 6: Run focused tests**

  Run: `python -m pytest tests/sft/test_qwen35_legacy_experiment.py tests/sft/test_launcher.py tests/verl_patches -q`

  Expected: all pass on Linux; local Windows patch-command limitations are recorded but do not waive remote Linux verification.

### Task 2: Periodic training monitor and checkpoint certification

**Files:**
- Create: `script/verl/sft/watch_training.py`
- Create: `tests/sft/test_watch_training.py`

**Interfaces:**
- Consumes: producer PID, log path, checkpoint root, status path, polling interval, expected mode, and optional timeout.
- Produces: atomic JSON states `running`, `completed`, or `failed`; returns zero only when a finite-loss log and required checkpoint artifacts are present.

- [ ] **Step 1: Write failing monitor tests**

  Test completion with a synthetic adapter checkpoint, failure after producer exit without checkpoint, failure on NaN/Inf loss or traceback, and stable atomic status replacement.

- [ ] **Step 2: Run tests and confirm the module is missing**

  Run: `python -m pytest tests/sft/test_watch_training.py -q`

  Expected: collection failure for `script.verl.sft.watch_training`.

- [ ] **Step 3: Implement checkpoint inspection and status writing**

  Parse finite losses from the log, reject explicit traceback/OOM/NaN markers, inventory checkpoint files and hashes, and atomically write status records. Poll by PID at the requested interval without depending on an SSH session.

- [ ] **Step 4: Run monitor and related watcher tests**

  Run: `python -m pytest tests/sft/test_watch_training.py tests/analysis/test_watch_evaluation.py -q`

  Expected: all pass.

### Task 3: Remote VeRL environment and real-data seam smoke

**Files:**
- Runtime only: `/root/autodl-fs/venvs/dcg-sft-qwen35-verl09/`
- Runtime only: `/root/autodl-fs/runs/dcg-shougang-5415-sft-20260917/`

**Interfaces:**
- Consumes: persistent source checkout, Qwen3.5 model, baseline SFT release, and tracked patch bundle.
- Produces: environment manifest, passed patch report, real-row dataset seam report, and staged fast-disk inputs.

- [ ] **Step 1: Synchronize only reviewed code and tests**

  Transfer the experiment CLI, monitor, tests, plan/spec, and existing baseline changes; verify archive SHA-256 before extraction.

- [ ] **Step 2: Create the isolated persistent environment**

  Reuse the host's CUDA-enabled PyTorch where compatible, install VeRL 0.9.0 and its SFT dependencies without storing credentials, and record exact package versions.

- [ ] **Step 3: Apply and verify required patches**

  Apply the tracked patch bundle with debug excluded and run the patch verifier. Stop on any target hash mismatch.

- [ ] **Step 4: Run repository and focused Linux tests**

  Run the focused SFT/patch tests and then the complete repository suite. Expected: zero failures.

- [ ] **Step 5: Validate real parquet through the patched dataset seam**

  Load one real Stage-1 and one real Stage-2 row using the actual VeRL dataset implementation and Qwen3.5 tokenizer. Assert rendered length below 4,096, non-empty assistant loss masks, and Stage-2 answer-mask coverage.

### Task 4: GPU smoke, formal training, and timed supervision

**Files:**
- Runtime only: smoke/formal configs, logs, statuses, and checkpoints under the SFT run root.

**Interfaces:**
- Consumes: verified environment, fast-disk model/data, and the Task-1 launcher.
- Produces: a two-step smoke checkpoint followed by a one-epoch formal LoRA checkpoint, each independently monitored.

- [ ] **Step 1: Materialize deterministic smoke parquet**

  Select a fixed ID-sorted subset containing both stages and write train/validation smoke files with their source IDs and hashes.

- [ ] **Step 2: Run the dry-run and two-step smoke**

  Verify the resolved command, launch with `nohup`, attach the monitor at a short interval, and require finite loss plus a valid adapter checkpoint.

- [ ] **Step 3: Adjust only memory realization if required**

  If and only if smoke OOMs, lower per-GPU micro-batch and increase gradient accumulation so effective batch size remains 32. Record the failed and successful configurations separately.

- [ ] **Step 4: Start the one-epoch formal run**

  Launch the exact approved configuration in a fresh formal output directory and attach `watch_training.py` at a 60-second interval. Record PID, config hash, data hashes, environment manifest, and status path.

- [ ] **Step 5: Certify completion before merge**

  Require monitor status `completed`, finite losses, expected checkpoint metadata, and adapter tensor hashes. A running formal job is reported as running, not as a completed SFT model.

### Task 5: Merge, fixed evaluation, and baseline comparison

**Files:**
- Runtime only: merged HF model, merge report, smoke/full evaluation reports, comparison JSON/Markdown.

**Interfaces:**
- Consumes: certified formal LoRA checkpoint and frozen test assets.
- Produces: loadable merged model and a certified 542-source SFT-vs-base comparison.

- [ ] **Step 1: Merge the certified LoRA checkpoint**

  Run `script.verl.sft.merge_lora_checkpoint` with the exact base model and verify the merge report plus all indexed model shards.

- [ ] **Step 2: Run two-source post-merge inference smoke**

  Require strict Stage-1 and Stage-2 JSON contracts before the full evaluation.

- [ ] **Step 3: Run the frozen 542-source evaluation in the background**

  Use the same data/assets, greedy decoding, thinking-disabled prompt rendering, `max_new_tokens=128`, and seed 42 as the non-training baseline; attach the existing evaluation watcher.

- [ ] **Step 4: Compare against the frozen baseline**

  Report protocol validity, recall@5, category EM/Macro-F1, level EM/Macro-F1, strict joint EM, failures, hashes, and deltas from 96.13%, 47.42%, 34.32%, 25.09%, and 7.93%, respectively.
