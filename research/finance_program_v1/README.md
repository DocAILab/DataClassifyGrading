# Finance Program: non-training + first Planner SFT

This directory contains an isolated code snapshot of the non-training baseline, teacher trajectory collection, first student SFT and paired evaluation. It does not replace the repository application.

## Layout

- `nontraining/`: Program controller, Skill contracts, execution, batch runner and audit.
- `teacher/`: Bailian `qwen3.8-2.4t-a95b` trajectory collection, bounded transport retry and budget accounting. See [teacher/README.md](teacher/README.md).
- `sft/`: first Qwen3.5-9B student SFT dataset extraction, Program+EOS-only encoding, completion loss, LoRA training and stage gates.
- `evaluation/`: matched base/SFT evaluation, adapter routing, controller replay, metrics and training verification.
- `tools/`: trajectory redaction utilities.
- `publication_tests/`: redaction regression tests.

Configuration examples, synthetic test fixtures and runtime Skill Markdown are part of the code snapshot. Reports, result summaries and trajectory data are not included.

## CPU tests

Run each stage in a separate process from its own directory; some module names are intentionally stage-local:

```powershell
cd nontraining
python -B -m pytest tests -q -p no:cacheprovider
cd ../teacher
python -B -m pytest tests -q -p no:cacheprovider
cd ../sft
python -B -m unittest discover -s tests -v
cd ../evaluation
python -B -m unittest discover -s tests -v
cd ..
python -B -m unittest discover -s publication_tests -v
```

Python 3.11+ is expected. CPU tests use synthetic inputs and mocked transport; they do not launch real training or API requests. Some CPU tests require PyTorch. GPU execution requires a compatible model environment; the original student run used Torch 2.9, Transformers 5.10.4, PEFT 0.21 and Accelerate 1.15.

## Running with authorized local inputs

Entry points include `nontraining/batch_runner.py` and `batch_audit.py`; `teacher/collect.py`; `sft/export_dataset.py`, `token_audit.py`, `train_student.py` and `pipeline.py`; and `evaluation/evaluate.py`, `replay.py`, `watch.py` and `verify_training.py`.

The teacher is Bailian `qwen3.8-2.4t-a95b`; the student is Qwen3.5-9B. Teacher weights are not trained by the student SFT code. Student supervision uses executed Planner Programs and EOS, not generated Skill replies or raw reasoning.

Real execution requires authorized local datasets, registry/corpus, model weights, stage manifests and resource configuration. Historical orchestration and extraction scripts expect a controlled delivery layout and are not a turnkey deployment. Prepare a new locally owned configuration/manifest; do not reuse an immutable completed run directory. No actual deployment manifest or private transport helper is shipped.

`evaluation/data/study.json` is a fictional synthetic fixture, not the real evaluation dataset. Adapter fixtures contain architecture/configuration, not model weights. Original datasets, raw trajectories, gold labels, model weights, credentials and lab endpoints are excluded. Cloning/importing this code does not launch jobs. Never store API keys in configuration, command arguments or logs.
