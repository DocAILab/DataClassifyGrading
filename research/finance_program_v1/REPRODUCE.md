# Reproduction and publication boundaries

## CPU tests

Run stages in separate processes from their own directories; names such as `gate` and `io_utils` are intentionally local:

```powershell
cd nontraining
python -B -m pytest tests -q -p no:cacheprovider
cd ../sft
python -B -m unittest discover -s tests -v
cd ../evaluation
python -B -m unittest discover -s tests -v
cd ..
python -B -m unittest discover -s publication_tests -v
```

Python 3.11+ is expected. The original publication checks used Python 3.13 with Torch 2.9 CPU operations. The actual 9B experiment used Torch 2.9 / Transformers 5.10.4 / PEFT 0.21 / Accelerate 1.15. These CPU tests do not download models or start GPU or teacher API work. The result update does not alter the original experiment code or its synthetic fixtures.

`evaluation/data/study.json` is a fictional field/category fixture, **not the real Val422 dataset**. The private-data prompt integration test and private transport/deployment/download helpers are excluded. Adapter fixtures contain architecture/configuration only, with a placeholder base path.

## Full financial Val422 results

The current [aggregate result](results/evidence-summary.json) covers all 422 unique sources and 844 formal trajectories, comparing an untrained Planner and the accepted first SFT A under the same F66 Program/Skill interface. Both arms call Skills; `base` is not a direct-answer Base-LLM. Model Skills always use the original base without adapter; only the SFT Planner enables the accepted LoRA.

The already verified subset was reused byte-for-byte and all remaining sources were run once under the same frozen protocol. Original smoke trajectories are excluded from formal metrics. The source union is exact and disjoint; failures and nontermination remain in each fixed 422 denominator. Labels were used only locally after generation and replay, never uploaded or inserted into model messages. Separate generation-stage walls are recorded, not represented as one full-run wall measurement.

All remote stages, controller/tokenizer/routing replay and archive steps exited successfully. On Windows the original and supplementary archives were safely extracted and all listed member hashes rechecked; full controller replay ran from each extracted root. The local replay does not claim a local real tokenizer: it is bound to the same-SHA remote real-tokenizer proof. All 844 formal trace hashes, per-ID metrics, paired counts and Macro-F1 were independently rechecked before publication.

Full numerical reproduction requires controlled input study/registry, exact source partition and order, local-only gold, full raw traces, original model revision and accepted first LoRA. These are not public. The `evaluation/` directory retains the original evaluation/replay code snapshot; it is not a self-contained launcher with private full-dataset orchestration or production data. The public JSON and trajectory views are a result release, not a replacement deployment bundle.

For controlled real execution, create a new owned configuration/manifest with authorized assets and the prescribed resource/identity checks. Do not overwrite immutable runs, silently change budgets/labels or infer permission for GPU/API work from this publication.

## Teacher and student trajectories

The current release contains **1,044 lossy projections**: unchanged teacher 200, matched full-validation base 422 and SFT 422. The teacher is **Bailian `qwen3.8-2.4t-a95b`**, not a locally trained 9B teacher. The first student uses 200 teacher-selected sources for Planner-only Program+EOS SFT; Skill output text and evaluation labels are not supervision.

Both student files align on public IDs `val-0001` through `val-0422`. No private source-ID mapping is published. The original trace SHA identifies controlled evidence; it is not the SHA of the redacted row. [trajectory-export.json](results/trajectory-export.json) records public-file sizes, row counts and SHA values, including the unchanged teacher file.

Actual Program structure, attempts, execution statuses and final classification/grade are retained. Prompts, raw reasoning, source metadata and semantic observation values are omitted. Literal arguments are replaced by `[REDACTED]` and flagged; unparseable replies retain status/digest, not raw text. No failed row is removed. This format cannot independently establish semantic correctness and must not be used as training data or full replay input.

The prior partial-validation result tables and standalone partial trajectory files are no longer part of the current branch tree. Original raw evidence and its cryptographic provenance remain preserved privately; the first SFT weights, teacher trajectories and experiment source code are unchanged.

## Interpretation

Val422 is a development set with eight supported classes out of twelve, one seed and potentially correlated table sources. It is not held-out Test or independent generalization. Supported-class and fixed-twelve-class Macro-F1 are reported separately; legal termination is not business correctness. Historical non-training interfaces are not interchangeable with this matched F66 comparison.
