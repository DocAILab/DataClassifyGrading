# Phase 1：无损 canonical standard 层（设计说明 + 迁移 note）

Status: 2026-08-20。只读建立数据事实层，**不改任何现有训练行为**；不推断
L1–L4 语义；不跨数据集假设同义；不修改样本、prompt、parser、reward、SFT/RL
parquet；`standard_data_level` 仅作标准参考，绝不覆盖 processed/canonical 的
sample `data_level`。

相关文件：
- Phase 0 语义：`docs/design/data_level_design.md`
- 标准构建实现：`src/agent/standards/`（contracts/sources/build/align）
- CLI：`script/standard/cli.py` → `python -m script.standard.cli`
- 产物：`data/standards/*.standard.json`、`artifacts/generated/provenance/`

---

## 1. canonical standard schema

每个标准 category（`data/standards/<ds>.standard.json` 内）：

```jsonc
{
  "category_id": "A1-1-1 | finance:客户.个人.个人基本概况信息",
  "name": "科研设备预约管理",
  "path": ["研发数据域", "产品研发", "科研检验", "科研设备预约管理"],  // 真实源层级深度，空层省略，不人为补空层
  "description": "...",                       // 该叶子所在层级的定义说明（原样保留）
  "code": "A1-1-1" | null,
  "standard_data_level": "L1|L2|L3|L4|null",  // 规范化的标准等级；无法解析时为 null
  "raw_level": "2 | l | 3 4",                 // 原始值，审计用
  "content": "…数据资源说明…",                 // 可选额外标准文本
  "source": {"file": "…", "sheet": "…", "row": …}   // 可追溯到原始标准文件+行号
}
```

顶层：`dataset / id_strategy / standard_name / standard_source{file,sheet} /
fingerprint` + `categories[]`。`fingerprint` 为 `categories` 内容（不含
source 行号）的 sha256——同输入确定性一致，与输入顺序/行号无关。

`category_id` 延续现有稳定 identity：finance = `finance:{L1}.{L2}.{L4}`
（`level_3` 仅 provenance，与 `DatasetConfig.identity_fields` 一致）；shougang
= guanji code。已验证：finance 233 个标准 id == 现有 registry 233 个 id，零差异。

## 2. 各数据集 standard source 状态

| dataset | standard_source | 事实源文件 | 状态 |
| --- | --- | --- | --- |
| finance | `finance` | `data/raw/金融行业数据安全分类分级标准指南.xlsx`（sheet Table 1） | built（233 类） |
| shougang | `shougang` | `data/raw/关基-数据分类分级目录.xlsx`（sheet 数据分类分级） | built（234 类） |
| infra | `shougang`（复用，不复制维护另一套） | — | 复用共享标准（64/64 对齐） |
| pers_info | `null`（missing / unknown） | 无已确认标准 | **不生成虚假 standard_data_level** |

pers_info：仓库内无确认的分类分级标准；18 类 registry 维持当前
dataset-derived 行为，但**不伪装成 canonical standard**——不生成
`pers_info.standard.json`，不在 summary 中编造等级。

## 3. sample ↔ standard 对齐统计（严格 category identity 连接）

| dataset | total | resolved | matched | mismatched | standard_missing | unresolved | resolved_match_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| finance | 568 | 531 | 529 | 2 | 0 | 37 | 99.62% |
| shougang | 19,415 | 18,393 | 18,393 | 0 | 0 | 1,022 | 100% |
| infra | 64 | 64 | 64 | 0 | 0 | 0 | 100% |
| pers_info | — | — | — | — | — | — | 无标准 |

- finance 未解析 37 = `missing_leaf 34 + path_mismatch 3`（即既有 canonical
  resolution 认定的 37 条非训练样本，类别不在 registry 中；本阶段不改）。标准覆盖
  233 类中 213 类未被任何样本观测到（universe ≫ 观测叶）。
- shougang 未解析 1,022 = 数据侧 `level_4='——'` 占位样本（与目录中的 `——`
  不同义，见 §4）。
- 对齐只读：不修改 sample `data_level`，不自动修标签。

## 4. 发现的数据异常（只报告，不修复）

1. **finance 原始标准 2 处不可解析等级**：row 150 `市场营销信息（公开）` 原始值
   `l`（疑似 `1` 的笔误）、row 168 `客户及监管相关音影像信息` 原始值 `3 4`
   （歧义）。→ `standard_data_level=null` + build issue，不做猜测。
2. **shougang 目录中"三级即叶子"层级**：10 行使 四列为文字 `——`、叶子码在 三级
   （B1-2 合同归并、B1-5 合同跟踪、B3-3/4/5、B4-2、B6-2、C1-5、C4-1）。它们
   **不是占位符**，是真实类别；canonical standard 已按真实深度 3 层保存
   （不发明第 4 层）。这是 Phase 0 规则 7 的直接实例。
3. **数据侧 `——` 与目录侧 `——` 语义不同**：shougang 样本的 `level_4='——'`
   是不可训练占位标签；目录中的 `——` 是"该层无子结点"标记。二者分别处理，
   不相干。
4. **legacy 信息损失（dict vs 原始标准）**：
   - finance：`financial_standards_dict.json` 把真实 4 层 path 压成
     L1-L2-leaf（232/233 类丢了 三级子类 provenance 层）；`l`/`3 4` 原始值在
     dict 中保留为 `l级`/`3 4级`。
   - shougang：`guanji_dict.json` 只保留 `name（code）`+描述，**丢了全部 path**
     （registry path 为空）**和分级列**（无 class）；并**漏掉 B3-6 中厚板作业计划**
     （目录中真实存在，registry 233 vs 标准 234）。

## 5. 下一阶段（registry/corpus 接口变化）

当前：`raw standard(Excel) →(lossy) standards_map JSON → canonical_corpus →
LeafRegistry + Corpus`（`financial_standards_dict.json`、`guanji_dict.json`、
`src/agent/task/canonical_corpus.py` 均为**有损中间层**）。

建议迁移为：
```
raw standard(Excel)
  → canonical standard（本项目，无损：path/description/code/standard_data_level）
  → LeafRegistry + Corpus
```
为此后续需修改的接口（本阶段不实现，只列出）：
1. `src/agent/task/canonical_corpus.py` 的 `parse_financial_standard` /
   `parse_guanji_standard` 改为消费 `CanonicalStandard`（或标志位切换数据源），
   `path` 使用标准真实深度而非补空层。
2. registry 生成在钳制当前 233（finance）/233（shougang）兼容的同时，
   决策是否收养标准的第 234 类 B3-6（当前数据 0 样本，仅宇宙完整性问题）。
3. `corpus_to_mapping` 是否附带 `standard_data_level` 作为 Stage2 知识——
   **属于下一步 task contract 决策**（Phase 0 §5：形态 A vs B），本阶段不预置。
4. 现有 dict 产物降级为 `legacy/derived`：审计对照用，不再作事实源。

## 6. 测试

```
tests/standards/
  test_contracts.py      round-trip / normalize / fragment / determinism
  test_build_finance.py  无损 path、identity 规则、异常上报、确定性
  test_build_shougang.py code/name/path/level、三级叶、no_code、确定性
  test_align.py          对齐桶、不修改样本、路由（infra→shougang、pers_info→None）
  test_real_xlsx.py      真实 Excel + canonical 集成断言（raw 缺失时 skip）
```

结果：`pytest tests/standards` → **26 passed**（raw 文件存在时集成用例全跑）；
全仓 pytest 见 PR 附注（本阶段无训练链路改动，回归为预防性）。

产物可重生成：`python -m script.standard.cli`（拒绝无 `--overwrite` 覆盖；
先行全部构建/对齐、后写盘；输出 sort_keys + 类别按 id 排序，重复构建字节级一致）。
raw Excel 仍是唯一事实源，但不进入任何训练代码依赖。

**git 边界**：`data/standards/*.standard.json` 被 `.gitignore` 的 `/data/*` 排除，
与 `data/processed`、`data/canonical` 同属**可再生层**（依赖 `data/raw` 恢复）；
入库的是 `src/agent/standards/`、`script/standard/`、`tests/standards/`、
`docs/design/phase1_canonical_standard.md`（+ Phase 0 的 `data_level_design.md`）
以及 `artifacts/generated/provenance/`（对齐审计 + 构建 summary，未忽略）。
