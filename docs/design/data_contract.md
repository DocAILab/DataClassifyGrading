# Canonical Data Contract

Status: **frozen** (Stage 3C/3B). The canonical identity is the single source of
truth for registry, corpus, ground truth, SFT labels, evaluation and reward.

## 1. One identity: canonical `category_id`

- **`target.category_id` is the only canonical semantic ground-truth identity**
  across SFT and RL. It is unique within one dataset's `LeafRegistry`.
  Model-facing supervision/actions use derived choice ids (see
  `docs/design/prompt_interface.md`) — the model never generates canonical ids.
- `classification.level_1..level_4` are **provenance only** — never read as a
  label, never fallback. There is deliberately **no `level_4` fallback**.
- Prompt-facing **choice ids are a derived, decoupled view** (see
  `docs/design/prompt_interface.md`); they never leak back into canonical
  records or reward semantics.

## 2. Canonical sample schema (`data/canonical/<dataset>/all.json`)

Input sample keeps the original fields (`classification`, `metadata`,
`data_level`, `label_status`, …) untouched as provenance. The resolver
appends:

```jsonc
"resolution_status": "resolved",            // resolved | missing_leaf | path_mismatch | ...
"target": {
  "leaf_level": "level_4",
  "leaf_name": "合同归并",                    // display name; not guaranteed unique
  "category_id": "B1-2",                     // the ONLY training identity (registry-unique)
  "category_path": ["生产数据域", "生产合同（订单）", "合同归并"]   // provenance, not an ID
}
```

## 3. Canonical corpus schema (`cfg/task/corpus/<dataset>.corpus.json`)

```jsonc
{ "categories": [ {
  "category_id": "A1-1-1",
  "name": "科研设备预约管理",
  "description": "…",            // primary description; may be empty
  "descriptions": ["…"],          // extra description documents (semantically distinct from examples)
  "path": ["研发数据域", "科研设备管理", "科研设备预约管理"],   // may be empty
  "code": "A1-1-1",               // may be empty; opaque stable id, digit groups are NOT level_2/3
  "examples": ["设备预约信息", "审核表"]
} ] }
```

- A category may have multiple descriptions/examples or none.
- **Production invariant**: the corpus must cover every registry category so
  Stage 2 can always resolve candidates by `category_id` (no registry fallback,
  validated by `require_corpus_covers_registry`).

## 4. Leaf registry (`cfg/task/registry/<dataset>.registry.json`)

- The `LeafRegistry` is the **complete leaf universe** of the classification
  standard/corpus, built from the corpus — **never derived from training
  samples** (samples may cover only a subset).
- Per-dataset registry sizes: finance **233** / infra **233** / shougang **233**
  / pers_info **18**; corpus sizes match.

## 5. Per-dataset identity strategy

| dataset | strategy | category_id shape |
|---|---|---|
| shougang | code (guanji) | `A1-1-1`, `B1-2`; letter↔level_1 100% (A/B/C); digits are opaque ordinals |
| infra | code, `registry_source=shougang` | shares shougang codes (infra ⊂ shougang) |
| finance | path (L1/L2/L4) | `finance:业务.账户信息.基本信息`; level_3 is provenance, not part of identity |
| pers_info | path (level_4 only) | `pers_info:学籍管理信息`; single-level labels |

## 6. Training entry rule (SFT + RL)

Only records with `resolution_status == "resolved"` **and** `target.category_id ∈
LeafRegistry` **and** `target.leaf_name == registry.get(category_id).name`
enter training; violations fail fast (no silent skip). Split membership follows
the original `data/processed/<dataset>/{train,val,test}.json` by record id. `canonical`
resolved / trainable counts (trainable = resolved within split boundaries):

| dataset | canonical resolved | trainable | outside splits |
|---|---|---|---|
| finance | 531 | 529 | 2 |
| infra | 64 | 64 | 0 |
| pers_info | 176 | 176 | 0 |
| shougang | 18,393 | 18,393 | 0 |

## 7. Consumers & files

- Contracts: `src/agent/task/contracts.py` (`LeafCategory`, `CorpusCategory`,
  `SampleTarget`, `LeafRegistry`, `TaskConfig`)
- Resolver: `src/agent/task/resolver.py`, identity: `src/agent/task/identity.py`,
  dataset config: `src/agent/task/dataset_config.py`
- Shared training helpers: `src/agent/training/common.py`
  (`canonical_target`, `build_candidates`, `require_corpus*`)
- Data-side alignment facts: `docs/data_alignment.md`; generated report:
  `artifacts/generated/alignment/data_alignment_report.{json,md}`
- Historical design records: `docs/phase_reports/stage2_contract_design.md`,
  `docs/phase_reports/stage3c_report.md`
