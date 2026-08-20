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

每个 standard entry（`data/standards/<ds>.standard.json` 内，`entries[]`，
一条原始标准行 = 一个 entry，**事实层不做任何聚合**）：

```jsonc
{
  "standard_entry_id": "finance:业务.合约协议.贷款业务信息.基本信息 | A1-1-1",
  "category_id": "finance:业务.合约协议.基本信息 | A1-1-1",
  "name": "基本信息",
  "path": ["业务", "合约协议", "贷款业务信息", "基本信息"],  // 真实源层级深度，空层省略
  "description": "…",
  "code": null | "A1-1-1",
  "standard_data_level": "L1|L2|L3|L4|null",
  "raw_level": "2 | l | 3 4",
  "content": "…数据资源说明…",
  "source": {"file": "…", "sheet": "…", "row": …}
}
```

顶层：`dataset / id_strategy / standard_name / standard_source{file,sheet} /
fingerprint / entries[] / training_projection`。

- **`standard_entry_id`** = 原始标准中的**真实身份**（finance 含真实三级子类
  `finance:{L1}.{L2}.{L3}.{L4}`；shougang = guanji code）。唯一。
- **`category_id`** = 当前训练/registry 兼容 alias（finance = L1-L2-leaf，与
  `DatasetConfig.identity_fields` 一致）。**不唯一**：多个标准 entry 可投影到同一
  训练类别。
- **`training_projection`** = 派生视图 `{category_id: [standard_entry_id…]}`，
  把 237 个 finance entry 投影到 233 个训练类别（237→233 是显式投影，**不是**
  事实层丢信息）。
- `fingerprint` = entries（不含 source 行号）+ projection 内容的 sha256；
  **与输入顺序无关**（每个 entry 原样保留、按 `standard_entry_id` 排序，
  不再依赖"首见顺序"）。

## 2. 各数据集 standard source 状态

| dataset | standard_source | 事实源文件 | 状态 |
| --- | --- | --- | --- |
| finance | `finance` | `data/raw/金融行业数据安全分类分级标准指南.xlsx`（sheet Table 1） | built（**237 entries / 233 training categories**） |
| shougang | `shougang` | `data/raw/关基-数据分类分级目录.xlsx`（sheet 数据分类分级） | built（234 entries） |
| infra | `shougang`（复用，不复制维护另一套） | — | 复用共享标准（64/64 对齐） |
| pers_info | `null`（missing / unknown） | 无已确认标准 | **不生成虚假 standard_data_level** |

pers_info：仓库内无确认的分类分级标准；18 类 registry 维持当前
dataset-derived 行为，但**不伪装成 canonical standard**——不生成
`pers_info.standard.json`，不在 summary 中编造等级。

## 3. sample ↔ standard 对齐统计（严格 category alias 连接）

| dataset | total | resolved | matched | mismatched | standard_missing | unresolved | resolved_match_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| finance | 568 | 531 | 529 | 2 | 0 | 37 | 99.62% |
| shougang | 19,415 | 18,393 | 18,393 | 0 | 0 | 1,022 | 100% |
| infra | 64 | 64 | 64 | 0 | 0 | 0 | 100% |
| pers_info | — | — | — | — | — | — | 无标准 |

- finance 未解析 37 条附 **`unresolved_evidence`**（evidence-only）：每条含
  `{status, leaf_name, candidate_standard_categories[]}`（同名校对候选，不修复）；
  其中 `missing_leaf 34 + path_mismatch 3`。
- 对齐支持多 entry 类别：sample 等级命中该类别任一 entry 的等级即 matched。
- 对齐只读：不修改 sample `data_level`，不自动修标签。

## 4. 发现的数据异常（只报告，不修复）

1. **finance 原始标准 2 处不可解析等级**：row 150 `市场营销信息（公开）` 原始值
   `l`、row 168 `客户及监管相关音影像信息` 原始值 `3 4`。→ `standard_data_level=null`
   + build issue，不做猜测。
2. **shougang 目录中"三级即叶子"层级**：10 行 四列为 `——`、叶子码在 三级
   （合同归并 B1-2、合同跟踪 B1-5 等）。是真实类别，已按真实深度 3 层保存
   （不发明第 4 层）。
3. **数据侧 `——` 与目录侧 `——` 语义不同**：样本 `level_4='——'` 是不可训练
   占位；目录 `——` 是"该层无子结点"。分别处理。
4. **legacy 信息损失**：finance dict 压平 4 层 path（232/233 类丢 三级 provenance
   层）；shougang dict 丢全部 path+分级，并**漏掉 B3-6 中厚板作业计划**。
   canonical standard 均已恢复。
5. **finance 5 条同训练类别的标准 entry**（业务/合约协议/基本信息 下的 合同通用/
   贷款业务/中间业务/资金业务/其他支付业务）全部保留，等级一致（2），经
   `training_projection` 投影为 1 个训练类别——这是低损事实层与训练投影的边界，
   不是 bug。

## 5. 下一阶段（registry/corpus 接口变化）

当前：`raw standard(Excel) → canonical standard（无损，本文档）→（下一步）LeafRegistry + Corpus`。

下一步需明确：
1. `src/agent/task/canonical_corpus.py` 改消费 `CanonicalStandard`：按
   `category_id`（training 投影）建 LeafRegistry，`path` 用标准真实深度；
   233 类保留、B3-6 是否收养（当前数据 0 样本，纯宇宙完整性）。
2. `corpus_to_mapping` 是否附带 `standard_data_level`/`standard_entry_id`
   作为 Stage2 知识——属任务契约决策（Phase 0 §5 A/B），本阶段不预置。
3. 现有 dict 产物降级为 legacy/derived 审计对照，不再作事实源。

## 6. 测试

```
tests/standards/
  test_contracts.py     round-trip / normalize / fingerprint / projection
  test_build_finance.py 无损 237、三级叶保持、level 异常、同类别多 entry determinism、reader issues
  test_build_shougang.py code/path/level、三级叶、no_code、reader issues、确定性
  test_align.py         对齐桶、多 entry 类别、不修改样本、unresolved evidence、路由
  test_checksum.py      checksum manifest 校验（缺失/不符）
  test_real_xlsx.py     真实 Excel + canonical 集成断言（raw 缺失时 skip）
```

结果：`pytest tests/standards` → **37 passed**；全仓见 PR。

产物可重生成：`python -m script.standard.cli`（拒绝无 `--overwrite` 覆盖；
先行全部构建/对齐、后写盘；重复构建字节级一致）。

### restore / 分发（Blocker-2 选型：B）

- **事实源不进入 Git**：raw workbook 属于数据提供方（含首钢内部目录），gitignored。
- **受控恢复流程**：从数据提供方/私有 artifact 取回两张表到 `data/raw/` 后，CLI
  先按 `script/standard/checksums.json`（已入库）校验 sha256，不符即拒绝构建
  （`--skip-checksum` 供离线调试显式绕过）。
- **git 边界**：`data/standards/*.standard.json` 仍被 `/data/*` 排除（可再生层，
  与 processed/canonical 一致）；入库的是 `src/agent/standards/`、`script/standard/`
  （含 `checksums.json`）、`tests/standards/`、`docs/design/*.md` 与
  `artifacts/generated/provenance/`。fresh clone 后按 restore 流程即可重建完全一致的
  事实层（fingerprint 可核对）。
