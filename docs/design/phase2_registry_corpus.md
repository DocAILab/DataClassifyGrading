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
  → cfg/task/corpus/<ds>.corpus.json   (Stage-2 知识 + 完整标准事实)
  → cfg/task/registry/<ds>.registry.json (Stage-1 宇宙，字段受限)
```

驱动：`python -m script.canonical.cli`（读 `--standard-dir` 默认
`data/standards`；finance←finance.standard.json；shougang/infra←shougang.standard.json；
pers_info←dataset universe fallback）。legacy 的 `financial_standards_dict.json` /
`guanji_dict.json` 不再被正式构建读取（降级 audit/legacy）。

## 2. registry / corpus schema 变化

**LeafRegistry（Stage1，只保留所需字段）**：`category_id / name / path / code` +
description。与 Phase 2 前 **逐字段一致**（id 集合、顺序、name、path、code 全部相同），
保证 Stage1 choice id 与 prompt 逐字节不变。唯一允许的差异：finance 1 条 description
为 whitespace-only（见 §6）。

**Corpus（Stage2 知识 + 完整标准事实）**：原字段（category_id/name/description/
descriptions/path/code/examples）保持不变（Stage2 prompt 可见面不变），**新增**：

```jsonc
{
  "standard_entry_ids": ["finance:业务.合约协议.合同通用信息.基本信息", "…"],
  "standard_entries": [{
    "standard_entry_id": "…",
    "name": "基本信息",
    "path": ["业务","合约协议","贷款业务信息","基本信息"],   // 真实源深度
    "description": "…",                                    // 该 entry 自己的内容
    "raw_level": "2",
    "standard_data_level": "L2",                           // 事实层，不进 prompt
    "raw_fields": { "level_2_definition": {…} }            // 层级定义（带 provenance）
  }],
  "scoped_annotations": [{ annotation_id, type, text, source_cell,
                           merged_range, start_row, end_row,
                           applies_to_standard_entry_ids }]
}
```

`CorpusCategory`（agent.task.contracts）新增 `standard_entry_ids` /
`standard_entries`（StandardEntryView）/ `scoped_annotations`
（CorpusScopedAnnotation）；round-trip 经 `load_corpus_categories` 完整还原。
build_report 新增 `standard_entries_out` / `excluded_categories` / `standard_name`；
`source` = 立即输入的标准文件（raw origin 在其内部 `standard_source`）。

## 3. finance 多 entry 处理（237 → 233，零丢失）

- 每 category_id 聚合全部 entry：**不取第一条**。`standard_entries` 保留 5 条
  （合同通用/贷款/中间/资金/非银行支付）各自的 id/真实 path/description/等级/raw_fields；
  `standard_entry_ids` 按源行序给出投影关系。
- Stage2 可见面（name + primary description + descriptions 追加）按**源行序**聚合，
  与旧 corpus 逐字节一致（5 桶的 primary=row56 合同通用、其余 4 条进 descriptions）。

## 4. B3-6（shougang 234 vs 233）

- standard 234 entries；active registry/corpus = 233；`build_report.excluded_categories
  = ["B3-6"]` + `category_excluded` issue（中厚板作业计划不在旧 registry，本阶段**不静默
  加入**以免改变 Stage1 宇宙；也不从标准删除）。infra 复用同一套（registry_source=shougang）。

## 5. legacy 脱离

`script.canonical.cli` 不再 import/读取 standards_map dict；parse_financial_standard /
parse_guanji_standard 保留为 legacy/audit 函数（测试仍覆盖），正式构建链不依赖。
corpus `source` 指向 `data/standards/*.standard.json`。

## 6. 已知残差（文档化，非静默）

finance `经营管理/运营管理/档案资料管理信息` 的 description：源标准单元格含一个多余
空格，legacy dict 曾静默删除。canonical standard 是事实源 → corpus/registry 保留源文本
（whitespace-only 差异 1 处；Stage1 不用 description，Stage2 该条描述多一空格）。
不复制 legacy 的单点编辑。已在 parity 测试中显式允许 whitespace-only 差异。

## 7. 复现与测试

- 复现：`python -m script.standard.cli --overwrite`（生成标准）→
  `python -m script.canonical.cli --overwrite`（生成 registry/corpus）。
- `pytest tests/task/test_canonical_corpus_phase2.py`：投影无损、多 entry round-trip、
  B3-6 报告、infra 复用、pers_info fallback、确定性、真实数据 parity（Stage1 字段、
  5 桶、43 条注解、B3-6 排除）。
- 全仓 `pytest` → 310 passed, 2 skipped（skip=本地无 verl，既有）。
