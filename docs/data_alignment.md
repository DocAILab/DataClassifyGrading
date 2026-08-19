# 数据对齐结论（dataset ↔ corpus/standard alignment）

> 人类维护的结论性文档。机器可复现的完整报告（含逐 dataset 覆盖率矩阵与
> UNKNOWN 明细）见 `artifacts/generated/alignment/data_alignment_report.{json,md}`，
> 由 `python -m script.analysis.analyze_dataset_corpus_alignment` 再生产。
> 本文件的内容更新需人工判断，不随脚本自动更新。

## 1. dataset ↔ 对应 corpus/standard

两种覆盖率口径（均来自生成报告，精确匹配/空白归一化）：

- **unique leaf coverage**：数据集唯一 leaf 中能在语料命中的比例（leaf 维度）；
- **sample leaf_exact coverage**：样本级精确匹配率（样本维度）。

| dataset | 对齐 source | unique leaf coverage | sample leaf_exact coverage | 备注 |
|---|---|---|---|---|
| finance | `corpus:finance` / `financial_standards_dict.json` | 20/25（80%） | 534/568（94.0%） | 语料 220 leaf ≫ 标注 25；fs 是 3 段路径（L1-L2-leaf），数据集是 4 级，**不把 join(level_1..4) 当身份** |
| infra | `standard:guanji_dict.json` | 4/4（100%） | 64/64（100%） | infra ⊂ shougang，共享 code 空间 |
| pers_info | `standard:education_dict.json` | 4/18（≈22%） | 35/176（19.9%） | 单级分类（仅 level_4）；14/18 leaf 无任何语料定义 |
| shougang | `standard:guanji_dict.json` | 192/193（≈99.5%） | 18393/19415（94.7%） | 样本口径下未匹配的 1022 条即 `——` 占位符（structural skip）；code 唯一、字母↔L1 100% |

> 注意：finance/shougang 的样本级匹配率（94.0%/94.7%）与 unique leaf
> 覆盖（80%/≈99.5%）分母不同（样本数 vs leaf 数），不要混读。

训练推荐 leaf level 全部为 **level_4**。

## 2. 现状数字（canonical 解析后，2026-08）

| dataset | 样本 | 完整 4 级路径 | level_4 leaf 数 | canonical resolved | trainable |
|---|---|---|---|---|---|
| finance | 568 | 381 | 25 | 531 | 529 |
| infra | 64 | 64 | 4 | 64 | 64 |
| pers_info | 176 | 0（单级） | 18 | 176 | 176 |
| shougang | 19,415 | 19,415 | 193 | 18,393 | 18,393 |

finance 有 2 条 resolved 记录落在原始 split 边界之外（数据管线事实，未静默补入）；
shougang `——` 占位符 1,022 条（5.3%）作为 structural skip 不进训练。

## 3. 需要记住的 schema 问题（人工视角）

- **finance**：187/568 样本 level_3 为空而 level_4 完整（路径不全，属数据事实）。
  2026-08-19 已修复 `"经营 管理"`→`"经营管理"` whitespace 变体（6 条），只读守护
  test 见 `tests/task/test_label_whitespace_variants.py`。
- **finance 命名噪声**：`交易清金额信息`（疑似错别字 vs `交易清结算信息`）、
  `基本信息（公开`（截断）等——标注侧定名前不改动。
- **cross-corpus leaf 冲突**：同名 leaf 出现在多个 standard（如 shougang
  `供应商管理数据` 在 guanji 与 shanghai_fta；finance `基本信息` 在
  financial_standards 与 education）——leaf 字符串不能作为身份，必须用
  canonical category_id。
- **pers_info**：单级分类 + 语料覆盖低（4/18），target 允许 description 缺失。
- **corpus:finance**：leaf-only（不带 path/code），历史遗留；正式 corpus 已
  改为 `CorpusCategory`（category_id/name/description/descriptions/examples）。
- **数据层级 vs data_level 字段**：data_level 分布与分类深度不符，语义未定
  （仅作 provenance，不解释为敏感级别）。

## 4. 仍需人工确认（摘要）

finance 5 个 leaf 无语料定义 / guanji code 数字段与 level_2/3 的对应 /
pers_info 14/18 语料来源 / education code 无层级可对照 / data_level 语义 /
shougang corpus 1 条 malformed 条目（`"nan"`）。完整清单见报告 §4.9。

## 5. 相关文档

- 身份策略与训练规则：`docs/design/data_contract.md`
- 历史设计记录：`docs/phase_reports/stage2_contract_design.md`、
  `docs/phase_reports/stage3c_report.md`
- 数据维护 SOP：`docs/新数据集运行说明.md`
