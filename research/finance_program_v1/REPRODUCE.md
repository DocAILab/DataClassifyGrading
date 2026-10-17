# Reproduction and publication boundaries

## CPU tests

Run each independent stage in a separate process from its own directory (module names such as gate/io_utils are intentionally local):

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

Python3.11+ is expected; publication checks ran Python3.13 with Torch2.9 CPU operations. The actual9B run used Torch2.9/Transformers5.10.4/PEFT0.21/Accelerate1.15. GPU execution and model imports require the matching environment; no dependency or model download occurs from these CPU tests.

The synthetic evaluation/data/study.json contains a fictional field/category. It is **not Val80**. The private-data initial-prompt integration test is not exported. Adapter inventory fixtures contain architecture/configuration only; the local base path is replaced with a placeholder. Copied source files and remaining tests retain their original behavior.

## Real experiment

Full numerical reproduction requires controlled teacher final selection/reviews/traces, planner200.jsonl, exact label-free study/Registry/fixed Val IDs, local-only gold, model revision and accepted LoRA. These are not public. SFT200 uses Program+EOS-only/no truncation; generated Skill text and future labels are not supervision.

Source entry points: nontraining/batch_runner.py and batch_audit.py; sft/export_dataset.py, token_audit.py, train_student.py and pipeline.py; evaluation/evaluate.py, replay.py, watch.py and verify_training.py. The teacher extractor expects a controlled sibling delivery layout; its known digest/path is provenance, not a public download mechanism.

A real run needs a new locally owned configuration/manifest and authorized input SHA/GPU UUID/shared-lock gate. Private transport/deploy/download helpers are excluded. No actual deployment manifest is shipped. Do not reuse an old immutable run directory, silently change labels/budgets or execute GPU/API work as part of inspecting this branch.

## Trajectories

440 lossy public projections are provided: teacher200, historical nontraining80, same-interface base80 and SFT80. Sequential public IDs replace source IDs; the three Val files align on the same public row order. No private ID mapping is published. Original trace SHA is a pointer to controlled evidence, not the SHA of the redacted row.

Planner Program structure, attempt/execution statuses and final result are preserved. Prompt messages, source metadata, raw reasoning and semantic observation values are omitted. Four same-interface base traces contain literal arguments that are explicitly redacted and flagged. Other public Program parameters remain exact. Invalid unparseable replies retain status and digest but not their raw text. No failed sample is removed.

These views must **not** be used as training data or full replay input. They do not independently establish semantic correctness. The full original controller replay was performed on private evidence before publication. Redaction code and regression tests are included; summary digests allow integrity checks of the public views.

## Interpretation

Historical P14/single-EOS nontraining results and matched grounded F66/dual-EOS base/SFT results are separate protocols. Only the latter supports the stated paired deltas. Val80 is a development set with8-of-12 class support, one seed and correlated sources; not held-out Test. Accuracy denominators remain80 even for nonterminal results.

