# Stage 1 BGE-M3 Retrieval Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and remotely validate a frozen BGE-M3 dense retrieval baseline that ranks all 193 level-4 labels from `field_name` alone and compares it fairly with character n-gram retrieval on 2,028 Shougang validation rows.

**Architecture:** Pure corpus, ranking, and metric modules remain dependency-light and testable with NumPy arrays. A lazy BGE-M3 adapter owns GPU encoding, while a val-only evaluator writes resumable per-row predictions and checksum-bearing reports. Both BGE-M3 and character n-gram scorers share the same stable ranking and metric implementation.

**Tech Stack:** Python 3.11, NumPy, PyTorch, FlagEmbedding `BGEM3FlagModel`, pytest, JSON/JSONL, Bash, Git bundle, RTX PRO 6000.

## Global Constraints

- Query input is exactly `metadata.field_name`.
- Candidate text comes only from the 193-category leaf registry; missing descriptions fall back to label names and are audited.
- Full evaluation reads only `val.json`; never enumerate, open, parse, upload, or read the real test split.
- Gold `classification.level_4` is used only after ranking for metrics.
- BGE-M3 uses normalized dense `[CLS]` vectors and inner product; sparse and ColBERT outputs remain disabled.
- Ties resolve by original registry order.
- Every report records `requested_splits=["val"]`, `real_test_split_read=false`, and `metadata_fields=["field_name"]`.
- Do not modify existing SFT/DPO behavior and do not push GitHub.

---

### Task 1: Deterministic Registry Corpus

**Files:**
- Create: `src/method/retrieval/__init__.py`
- Create: `src/method/retrieval/corpus.py`
- Test: `tests/method/retrieval/test_corpus.py`

**Interfaces:**
- Consumes: `src.agent.task.contracts.LeafRegistry`.
- Produces: `RegistryDocument(category_id, text, provenance, checksum)` and `build_registry_documents(registry)`.

- [ ] **Step 1: Write failing corpus tests**

```python
def test_registry_document_uses_description_and_audits_fallback(registry):
    docs = build_registry_documents(registry)
    assert docs[0].text == "标签名称：A\n标签描述：alpha data"
    assert docs[0].provenance == "registry_description"
    assert docs[1].text == "标签名称：B"
    assert docs[1].provenance == "label_name_fallback"
    assert len(docs[0].checksum) == 64
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/method/retrieval/test_corpus.py`

Expected: collection failure because `src.method.retrieval.corpus` does not exist.

- [ ] **Step 3: Implement immutable document rendering**

```python
@dataclass(frozen=True)
class RegistryDocument:
    category_id: str
    text: str
    provenance: str
    checksum: str

def build_registry_documents(registry: LeafRegistry) -> tuple[RegistryDocument, ...]:
    documents = []
    for category in registry.categories:
        description = " ".join(category.description.split())
        text = f"标签名称：{category.category_id}"
        provenance = "label_name_fallback"
        if description:
            text += f"\n标签描述：{description}"
            provenance = "registry_description"
        documents.append(RegistryDocument(category.category_id, text, provenance, sha256(text.encode()).hexdigest()))
    return tuple(documents)
```

- [ ] **Step 4: Run corpus tests and existing task-contract tests**

Run: `pytest -q tests/method/retrieval/test_corpus.py tests/task/test_contracts_and_prompts.py`

Expected: all pass.

- [ ] **Step 5: Commit the corpus unit**

```bash
git add src/method/retrieval tests/method/retrieval/test_corpus.py
git commit -m "feat: build audited retrieval corpus"
```

### Task 2: Stable Ranking and Retrieval Metrics

**Files:**
- Create: `src/method/retrieval/ranking.py`
- Create: `src/method/retrieval/metrics.py`
- Test: `tests/method/retrieval/test_ranking.py`
- Test: `tests/method/retrieval/test_metrics.py`

**Interfaces:**
- Consumes: finite NumPy score matrices, registry IDs, gold IDs, prediction rows.
- Produces: `stable_rank(scores, registry_ids, top_k)`, `build_prediction(...)`, and `summarize_retrieval(rows, registry_ids, train_counts)`.

- [ ] **Step 1: Write failing ranking tests**

```python
def test_stable_rank_breaks_equal_scores_by_registry_order():
    ranked = stable_rank(np.array([0.8, 0.8, 0.9]), ("A", "B", "C"), top_k=3)
    assert ranked == (("C", 0.9), ("A", 0.8), ("B", 0.8))

def test_stable_rank_rejects_non_finite_scores():
    with pytest.raises(ValueError, match="finite"):
        stable_rank(np.array([0.1, np.nan]), ("A", "B"), top_k=2)
```

- [ ] **Step 2: Run ranking tests and verify RED**

Run: `pytest -q tests/method/retrieval/test_ranking.py`

Expected: import failure for the missing ranking module.

- [ ] **Step 3: Implement minimal stable ranking and row construction**

```python
def stable_rank(scores, registry_ids, *, top_k):
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(registry_ids),) or not np.isfinite(values).all():
        raise ValueError("scores must be a finite vector aligned to registry IDs")
    order = np.lexsort((np.arange(len(registry_ids)), -values))[:top_k]
    return tuple((registry_ids[index], float(values[index])) for index in order)
```

- [ ] **Step 4: Write failing exact-metric tests**

```python
def test_metrics_compute_micro_macro_mrr_rank_and_coverage():
    report = summarize_retrieval(FIXTURE_ROWS, registry_ids=("A", "B", "C"), train_counts={"A": 3, "B": 1, "C": 0})
    assert report["recall_at_1"] == pytest.approx(0.5)
    assert report["recall_at_3"] == 1.0
    assert report["macro_recall_at_1"] == pytest.approx(0.5)
    assert report["mrr"] == pytest.approx(0.75)
    assert report["registry_top5_coverage_count"] == 3
```

- [ ] **Step 5: Run metric tests and verify RED**

Run: `pytest -q tests/method/retrieval/test_metrics.py`

Expected: import failure for the missing metrics module.

- [ ] **Step 6: Implement exact aggregate and per-class metrics**

Implement micro Recall@1/3/5, positive-support macro Recall@1/3/5, MRR, mean/median gold rank, per-class support, train-frequency buckets, registry Top-5 coverage, unique Top-5 tuples, and duplicate/OOV/invalid counts. Reject duplicate `source_id` values and gold labels outside the registry.

- [ ] **Step 7: Run focused and project regression tests**

Run: `pytest -q tests/method/retrieval/test_ranking.py tests/method/retrieval/test_metrics.py tests/evaluation/test_classification.py`

Expected: all pass.

- [ ] **Step 8: Commit ranking and metrics**

```bash
git add src/method/retrieval/ranking.py src/method/retrieval/metrics.py tests/method/retrieval/test_ranking.py tests/method/retrieval/test_metrics.py
git commit -m "feat: score Stage1 retrieval metrics"
```

### Task 3: BGE-M3 Dense Encoder and Character n-gram Control

**Files:**
- Create: `src/method/retrieval/bge_m3.py`
- Create: `src/method/retrieval/char_ngram.py`
- Test: `tests/method/retrieval/test_bge_m3.py`
- Test: `tests/method/retrieval/test_char_ngram.py`

**Interfaces:**
- Consumes: sequences of query/corpus strings.
- Produces: `BgeM3DenseEncoder.encode_queries`, `encode_corpus`, `dense_scores`, and `char_ngram_scores`.

- [ ] **Step 1: Write failing encoder-validation tests**

```python
def test_dense_scores_require_normalized_aligned_vectors():
    queries = np.array([[1.0, 0.0]])
    corpus = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert dense_scores(queries, corpus).tolist() == [[1.0, 0.0]]
    with pytest.raises(ValueError, match="normalized"):
        dense_scores(queries * 2, corpus)
```

- [ ] **Step 2: Run BGE tests and verify RED**

Run: `pytest -q tests/method/retrieval/test_bge_m3.py`

Expected: import failure for the missing BGE module.

- [ ] **Step 3: Implement lazy official encoder adapter**

```python
class BgeM3DenseEncoder:
    def __init__(self, model_path, *, device="cuda:0", batch_size=32, use_fp16=True):
        self._model = BGEM3FlagModel(str(model_path), devices=device, pooling_method="cls", use_fp16=use_fp16)

    def encode_queries(self, texts):
        return self._model.encode_queries(list(texts), return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"]

    def encode_corpus(self, texts):
        return self._model.encode_corpus(list(texts), return_dense=True, return_sparse=False, return_colbert_vecs=False)["dense_vecs"]
```

Import `FlagEmbedding` inside the constructor so pure unit tests remain usable without the optional package. Validate rank-2 shape, dimension agreement, finiteness, and unit L2 norms before matrix multiplication.

- [ ] **Step 4: Write failing lexical-control tests**

```python
def test_char_ngram_control_prefers_overlapping_chinese_label_text():
    scores = char_ngram_scores(["设备编号"], ["标签名称：设备基本资料", "标签名称：车辆信息"])
    assert scores.shape == (1, 2)
    assert scores[0, 0] > scores[0, 1]
```

- [ ] **Step 5: Run lexical tests and verify RED**

Run: `pytest -q tests/method/retrieval/test_char_ngram.py`

Expected: import failure for the missing control module.

- [ ] **Step 6: Implement deterministic character n-gram scoring**

Use the existing `_features` and cosine-overlap definition from DPO hard-negative retrieval, moved or wrapped without changing DPO behavior. Return a finite query-by-document matrix aligned to registry order.

- [ ] **Step 7: Run all retrieval tests**

Run: `pytest -q tests/method/retrieval`

Expected: all pass without loading BGE-M3.

- [ ] **Step 8: Commit scorers**

```bash
git add src/method/retrieval/bge_m3.py src/method/retrieval/char_ngram.py tests/method/retrieval/test_bge_m3.py tests/method/retrieval/test_char_ngram.py
git commit -m "feat: add BGE-M3 dense retrieval scorer"
```

### Task 4: Val-only Evaluation, Recovery, and Reports

**Files:**
- Create: `src/method/retrieval/evaluation.py`
- Create: `src/method/retrieval/script/__init__.py`
- Create: `src/method/retrieval/script/evaluate_stage1.py`
- Test: `tests/method/retrieval/test_evaluation.py`
- Test: `tests/method/retrieval/test_cli.py`

**Interfaces:**
- Consumes: input directory, registry, model path, output directory, method, batch size.
- Produces: audited JSON/JSONL artifacts, resumable per-row predictions, reports, and completion/failure markers.

- [ ] **Step 1: Write failing val-only and resume tests**

```python
def test_evaluator_reads_only_val_and_resumes_by_source_id(tmp_path, registry):
    write_json(tmp_path / "val.json", VAL_ROWS)
    result = evaluate_stage1(input_dir=tmp_path, registry=registry, scorer=FakeScorer(), output_dir=tmp_path / "out")
    assert result["requested_splits"] == ["val"]
    assert result["real_test_split_read"] is False
    assert result["metadata_fields"] == ["field_name"]
    assert result["rows"] == len(VAL_ROWS)
```

Add tests that reject duplicate source IDs, non-registry gold labels, stale embedding cache metadata, prediction/report disagreement, and any CLI split option other than the hard-coded val contract.

- [ ] **Step 2: Run evaluator tests and verify RED**

Run: `pytest -q tests/method/retrieval/test_evaluation.py tests/method/retrieval/test_cli.py`

Expected: missing module or missing interface failures.

- [ ] **Step 3: Implement evaluator and atomic artifacts**

Load only `input_dir / "val.json"` by explicit literal path. Build the corpus, cache embeddings with model/corpus/config checksums, batch-score unfinished source IDs, append JSONL rows atomically, summarize both methods, and verify 2,028 unique IDs before writing `COMPLETE`.

- [ ] **Step 4: Implement CLI with no split argument**

```python
parser.add_argument("--input-dir", type=Path, required=True)
parser.add_argument("--registry", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--limit", type=int)
```

`--limit` exists only for smoke and selects a deterministic prefix after loading val; it never changes the split contract.

- [ ] **Step 5: Run evaluator, CLI, and full local tests**

Run: `pytest -q tests/method/retrieval && pytest -q -m "not verl"`

Expected: all retrieval tests and all non-VeRL project tests pass.

- [ ] **Step 6: Commit evaluation pipeline**

```bash
git add src/method/retrieval tests/method/retrieval
git commit -m "feat: evaluate val-only Stage1 retrieval"
```

### Task 5: Recoverable Remote Launcher and Full Verification

**Files:**
- Create: `src/method/retrieval/script/run_stage1_bge_m3.sh`
- Create: `src/method/retrieval/script/start_stage1_bge_m3.sh`
- Create: `docs/STAGE1_RETRIEVAL.md`
- Test: `tests/method/retrieval/test_pipeline.py`

**Interfaces:**
- Consumes: remote paths via environment variables.
- Produces: smoke verification, full evaluation, status transitions, detached execution, and reproducible commands.

- [ ] **Step 1: Write failing launcher-contract tests**

```python
def test_pipeline_is_val_only_and_runs_smoke_before_full():
    launcher = PIPELINE.read_text(encoding="utf-8")
    assert "--limit 16" in launcher
    assert "stage1-bge-m3-v1" in launcher
    assert "test.json" not in launcher
    assert launcher.index("--limit 16") < launcher.index("FULL_COMPLETE")
```

Also assert detached `setsid`/`nohup`, status updates, model checksum capture, free-space preflight, and no GitHub push.

- [ ] **Step 2: Run pipeline tests and verify RED**

Run: `pytest -q tests/method/retrieval/test_pipeline.py`

Expected: missing launcher failure.

- [ ] **Step 3: Implement launchers and concise operator documentation**

The pipeline runs tests, verifies at least 8 GiB free space, verifies/downloads `BAAI/bge-m3`, runs a 16-row real-encoder smoke, verifies vector dimension/norm/schema, runs all 2,028 val rows, cross-checks reports, and writes `COMPLETE`. The detached launcher records PID, SID, code HEAD, and logs.

- [ ] **Step 4: Run fresh complete local verification**

Run: `pytest -q tests/method/retrieval && pytest -q -m "not verl" && git diff --check`

Expected: zero failures and clean diff checks.

- [ ] **Step 5: Commit remote launcher**

```bash
git add src/method/retrieval/script docs/STAGE1_RETRIEVAL.md tests/method/retrieval/test_pipeline.py
git commit -m "feat: run recoverable BGE-M3 baseline"
```

- [ ] **Step 6: Sync without pushing GitHub**

Create a Git bundle from `codex/stage1-bge-m3-v1`, upload it to the remote host, create `/root/autodl-tmp/worktrees/DataClassifyGrading-stage1-bge-m3`, and verify remote HEAD and clean status. Do not modify `/root/autodl-tmp/workspace`.

- [ ] **Step 7: Run remote tests and real smoke**

Run the existing Python environment, install only the compatible FlagEmbedding dependency if absent, download BGE-M3 under `/root/autodl-tmp/models/bge-m3`, and launch the 16-row smoke. Verify 1024-dimensional normalized vectors, finite scores, 16 unique source IDs, val-only audit, and GPU memory.

- [ ] **Step 8: Run and audit all 2,028 val rows**

Launch the detached full pipeline. Verify BGE-M3 and character n-gram row counts, unique IDs, Recall@1/3/5, macro metrics, MRR, gold-rank statistics, label coverage, diversity, no OOV/duplicates/invalid rows, runtime/GPU reports, checksums, disk usage, and `real_test_split_read=false`.

- [ ] **Step 9: Report the evidence-backed result**

Compare BGE-M3 against character n-gram and the previous generative Stage 1 Recall@5 of 13.4122%. State whether BGE-M3 should advance to the separate DPO Stage 2 end-to-end experiment; do not present Stage 1 Recall as end-to-end accuracy.
