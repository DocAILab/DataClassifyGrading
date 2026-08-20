# Phase 2：LeafRegistry / Corpus 正式从 CanonicalStandard 派生

Status: 2026-08-20。数据链统一为 `raw standard → CanonicalStandard → LeafRegistry + Corpus`；
legacy standards_map dict 脱离正式构建链（仅 audit）。**不修改 Stage1/Stage2 prompt、
parser、reward、SFT/RL**；`standard_data_level` 与 grading 注解只进 corpus 事实层，
本阶段不暴露给模型。

## 1. 新数据链

```text
raw standard (Excel, data/raw, gitignored)
  → CanonicalStandard (Phase 1, data/standards/*.standard.json, 可再生成)
  → build_from_standard()  (src/agent/task/canonical_corpus.py)
        + explicit projection policy
        (DatasetConfig.projection_excluded_category_ids)
  → cfg/task/corpus/<ds>.corpus.json   (Stage-2 知识 + 完整标准事实)
  → cfg/task/registry/<ds>.registry.json (Stage-1 宇宙，字段受限)
```

驱动：`python -m script.canonical.cli`（读 `--standard-dir` 默认
`data/standards`；finance←finance.standard.json；shougang/infra←shougang.standard.json；
pers_info←dataset universe fallback）。**正式构建输入只有 CanonicalStandard +
projection policy**：legacy 的 `financial_standards_dict.json` / `guanji_dict.json`
与旧 registry 均不参与构建，仅用于 audit / parity 测试。

## 2. registry / corpus schema 变化

**LeafRegistry（Stage1，只保留所需字段）**：`category_id / name / path / code` +
description。与 Phase 2 前 **逐字段一致**（id 集合、顺序、name、path、code 全部相同），
保证 Stage1 choice id 与 prompt 逐字节不变。

**Corpus（Stage2 知识 + 完整标准事实）**：原字段（category_id/name/description/
descriptions/path/code/examples）保持不变（Stage2 prompt 可见面不变），**新增**：

```jsonc
{
  "standard_entry_ids": ["finance:业务.合约协议.合同通用信息.基本信息", "…"],
  "standard_entries": [{
    "standard_entry_id": "…",
    "name": "基本信息",
    "path": ["业务","合约协议","贷款业务信息","基本信息"],   // 真实源深度
    "description": "…",                                    // 该 entry 自己的 内容/四级定义
    "content": "…",                                       // 数据资源说明（shougang）
    "code": "A1-1-1" | null,
    "raw_level": "2",
    "standard_data_level": "L2",                          // 事实层，不进 prompt
    "source": {"file": "…", "sheet": "…", "row": …},   // Phase-1 source provenance
    "raw_fields": { "level_2_definition": {…}, "resource": {…} }  // 层级定义/数据来源
  }],
  "scoped_annotations": [{ annotation_id, type, text, source_cell,
                           merged_range, start_row, end_row,
                           applies_to_standard_entry_ids }]
}
```

`StandardEntryView`（agent.task.contracts）携带 Phase-1 `StandardCategory` 的全部
事实字段（含 content / code / source / raw_fields），经 corpus JSON →
`load_corpus_categories()` 完整 round-trip 不丢字段。shougang 三字段各归其位：
`description`=四级定义，`content`=数据资源说明，`resource` 仍在 `raw_fields`。
build_report 新增 `standard_entries_out` / `excluded_categories` / `standard_name`；
`source` = 立即输入的标准文件。

## 3. finance 多 entry 处理（237 → 233，零丢失）

- 每 category_id 聚合全部 entry：**不取第一条**。`standard_entries` 保留 5 条
  （合同通用/贷款/中间/资金/非银行支付）各自的 id/真实 path/description/等级/raw_fields；
  `standard_entry_ids` 按源行序给出投影关系。
- Stage2 可见面（name + primary description + descriptions 追加）按**源行序**聚合，
  与旧 corpus 逐字节一致（5 桶的 primary=row56 合同通用、其余 4 条进 descriptions）。

## 4. B3-6（shougang 234 vs 233）与 projection policy

- **policy 位置**：`DatasetConfig.projection_excluded_category_ids`（dataset_config.py
  BUILTIN shougang = `("B3-6",)`；infra 经 `registry_source=shougang` 继承同一 policy；
  finance/pers_info 无排除）。
- 正式链：`CanonicalStandard + policy → registry/corpus`；`build_from_standard`
  默认读取 policy（也可显式传 `excluded_category_ids`）。排除的类别进
  `build_report.excluded_categories` + `category_excluded` issue —— 不静默加入/删除。
- 旧 registry **不再被正式 builder 读取**（`_previous_active_ids` 已删除）；
  仅 parity 测试用它核对"新结果与旧 Stage1 universe 一致"。

## 5. legacy 脱离

`script.canonical.cli` 不读取 standards_map dict 与旧 registry；
`parse_financial_standard` / `parse_guanji_standard` 保留为 legacy/audit 函数
（测试仍覆盖），正式构建链零依赖。corpus `source` 指向 `data/standards/*.standard.json`。

## 6. CI / fresh clone

- CLI 集成测试改用 **tmp canonical-standard fixture**（`--standard-dir` 指向临时目录），
  fresh clone / CI 无需私有 raw/standard 数据即可跑；真实 standard parity 测试在
  文件缺失时 skip。

## 7. 复现与测试

- 复现：`python -m script.standard.cli --overwrite`（生成标准）→
  `python -m script.canonical.cli --overwrite`（生成 registry/corpus）。
- `pytest tests/task/test_canonical_corpus_phase2.py` + `test_canonical_corpus.py`：
  fresh-clone CLI、policy 排除与报告、EntryView content/source/code round-trip、
  shougang 三字段区分、pers_info fallback、确定性、真实数据 parity。
- 全仓 `pytest` → 313 passed, 2 skipped（skip=本地无 verl，既有）。
