# Reproduction and publication boundaries

This release includes code, aggregate metrics and 397 lossy trajectory projections. It does not include controlled data/gold, private source-ID mappings, raw prompts/replies, tokenizer/model assets, credentials or historical machine paths.

## Offline checks

Python 3.11+; publication checks were run with Python 3.14 on Windows. Run these commands separately from this directory:

```text
python -B -m unittest discover -s code/tests -v
python -B -m unittest discover -s code/upstream/runtime/tests -v
python -B -m unittest discover -s publication_tests -v
python -B tools/verify_release.py
```

The tests use fictional fixtures and scripted/mock transports. They do not call live models or train. Windows supervisor/status operations require psutil. Actual API reference token counting additionally requires tokenizers and jinja2 and the separately authorized pinned reference tokenizer. Retained local GPU code is not evidence that this API experiment used local weights.

## Code provenance

`results/code-publication.json` binds all 145 files in the historical final runtime. The 117 vendored upstream files remain byte-identical to team commit `9473b2c2a330b7f48dc0c1831bda11e36d428379`.

The public copy changes exactly three historical files: `code/README.md`, the local credential-file default in `api_backend.py`, and the local tokenizer-path default in `night_prepare.py`. Environment variables VEHICLE_NONTD_CREDENTIAL_FILE and VEHICLE_NONTD_TOKENIZER accept local asset paths, not API keys. These publication changes do not change Planner/Skill prompts, execution budgets, inference policy or scoring. New manifests are required; the public copy must not be inserted into the immutable historical run. Original code hashes and private run/score/replay hashes are retained in the aggregate evidence.

## Controlled data and fresh runs

Obtain the authorized DocAILab/DCG dataset at revision `84a9a462dcb828332f980ef121f40e1281ce1794` independently. `code/pinned-inputs.json` provides five input file hashes. From `code/`, `prepare.py --source <controlled-dataset-root> --output <new-directory-outside-git>` builds allowlisted inference data and an evaluator-only gold file. The supplied tests exercise the same path with fictional inputs. Preparation does not start a model.

`runner.py` provides mock/local/API execution; `score.py` performs evaluator-only scoring; `replay.py` re-executes a recorded complete controller trace without a model. `night_prepare.py`, `night_runner.py`, `night_supervisor.py` and `Night-Control.ps1` contain the bounded Windows API orchestration and recovery checks. Help and import do not start paid work. Do not infer permission for a new live run from this publication.

Exact historical replay requires the original immutable runtime and manifests, controlled study/registry, full traces and call records. Independent numerical scoring additionally requires controlled gold. These are intentionally absent. Hosted model weight revision/precision is not exposed, so a new API run is not guaranteed to reproduce the same outputs.

## Public trajectory schema

Rows are `val-0001` through `val-0397` in fixed evaluation order. Only known tool names, integer budget/attempt counters, enumerated statuses, termination flags and opaque predicted category/grade aliases are retained. All arguments (including state references), messages, raw outputs, observations, summaries, source metadata and gold are omitted. Invalid Planner replies retain their attempt/status, never the raw text. A trace SHA binds each public row to privately retained evidence. Aliases are consistent within this release; no mapping to private source IDs or standard labels is published.

All 397 receipts are represented, including 17 backend errors and 265 nonterminal budget failures. Public projections allow auditing aggregate completion and tool-sequence behavior; they cannot independently establish semantic correctness or serve as full replay/SFT data.

`results/files.sha256.json` lists every publication file except itself. Compare these hashes after download. `results/publication-verification.json` records the actual local checks and privacy scan scope; no GPU/API experiment is part of publication.
