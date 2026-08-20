# data_level / classification / field_sensitive 角色区分（设计说明 · Phase 0）

Status: 只读记录（2026-08-20，数据快照 v1 后新增 raw Excel 已核对）。
原则：只记录仓库内可验证事实，不猜测等级语义；**不改任何代码 / 数据 / 现有训练行为**。

**Phase 0 冻结范围**：只冻结三个字段「是什么、来自哪里、当前代码是否训练」。**不决定** `data_level` 最终训不训练——那属于 Stage2 task contract 的决策，本文档保持开放。

## 1. 三个概念是什么（角色模型）

### `classification`（`level_1..level_4`）
- **角色**：sample label（源端标注），并且是**当前**的 training target（经 canonical 解析为 `target.category_id`，Stage1/Stage2 唯一监督目标）。
- **槽位口径（重要，避免概念混淆）**：`level_1..level_4` 是统一分类槽位。当前 processed 数据把**最终/最小粒度类别**放进 `level_4` 槽位，但不同源数据的**真实分类深度并不统一**：
  - pers_info：源数据是单层分类（学籍管理信息…），canonical schema 把它放进 `level_4`；`level_1/2/3` 为空，**不是说源标准真有四级体系**。
  - finance：四层填充（`level_1..level_4`）；shougang/infra：四层填充。
  - 因此**不能把 `level_4` 等价成"真实标准的第四层"**；建无损 standard（Phase 1）时按各源的真实深度建模。

### `data_level`（`L1..L4`）
- **角色**：**sample label——样本携带的分级标签**。是否同时可被确定为 standard knowledge，**分数据集**：
  - finance / shougang / infra：有证据可追溯到分类分级标准/目录（finance↔《金融行业数据安全分类分级标准指南》"最低安全级别参考"；shougang/infra↔首钢京唐数据分类分级目录"分级"列，192/192 零冲突）。
  - pers_info：**目前只有 sample label，尚无已确认的 standard knowledge 来源**（源 Excel 直接填 `L1/L2/L3`，仓库内无对应标准文档）。
  - 所以不做笼统定义"`data_level` = standard knowledge 的样本级表现"——pers_info 就是反例。
- **训练目标状态**：当前实现未进入 training target（`src/agent`、`script/verl`、`script/canonical` 对 `data_level` 零引用，parquet 不导出）；**目标设计倾向将 `data_level` 作为与 classification 并列的模型预测目标**，即模型需要根据字段语义、业务上下文和分级规则独立判断安全等级，而不是仅根据 category 查表返回标准等级。最终接口仍待 Stage2 task contract 冻结。
- **训练目标总口径**：
  ```text
  当前实现：
    classification → training target
    data_level     → provenance only
  目标任务方向：
    classification → 模型预测
    data_level     → 模型独立预测
    standard_data_level → 分级参考 / 审计依据，不直接等同于字段最终等级

  最终输出格式、prompt 可见信息和 reward
  → 待 Stage2 task contract 冻结
  ```

### `sample_data_level` 与 `standard_data_level`（两个变体，必须分开保存）
- **`sample_data_level`** = 原始数据中具体字段的实际分级标签（即 § 上文 `data_level` 的本义），作为训练/评测 gold 候选。
- **`standard_data_level`** = 分类分级标准给 category 的参考/最低等级（如 finance 标准列原名就叫**"最低安全级别参考"**）。
- 二者**必须分别保存，不应在 canonicalization 时相互覆盖**：finance 的"最低安全级别参考"本身就不适合直接建模成 `category → 唯一最终 data_level`。
- 后续 Phase 1 的 canonical standard 建议明确字段名 `standard_data_level`（值如 `"L3"`），**不要**简单命名为 `data_level`，否则日后容易再次与 sample gold 混淆。

### `field_sensitive`（是/否）
- **角色**：**独立 field-level 源端标注**，不属于上述两者。
- 仅存在于 `data/raw/关基设施数据分类分级-不包含训练-用于测试(1).xlsx`（infra 测试来源）的"字段是否敏感"列；mapping 未映射 → 未进入 processed/canonical/parquet。
- 与 `data_level` **不同步**（64 行：`L3+是=32 / L3+否=14 / L2+是=14 / L2+否=4`），证据明确，故为独立概念。

### 三者关系（已验证统计事实）
- `data_level` 与分类深度无关（pers_info 单层分类却分布 L1–L3）；在数据上 `data_level` 对叶子类别近似确定性映射（finance 18/20、infra 4/4、pers_info 17/18、shougang 192/192 类别为单一级别，全库仅 4 条跨级样本）。

## 2. 三个字段在管线各层的位置

| 层 | classification | data_level | field_sensitive |
| --- | --- | --- | --- |
| raw（data/raw/*.xlsx） | 源列：一级子类/一级分类/分类 等 | 源列：数据级别(finance) / 分级(infra,shougang,pers_info)，原始值 `1..4`（pers_info 为 `L1..L3`） | 仅 `关基设施…用于测试(1).xlsx` 有"字段是否敏感"列 |
| preprocessing（`script/preprocessing/processor.py`） | `normalize_label` 归一化（剥 `（A1-1-3）` 代码后缀） | `normalize_level`：`1/2/3/4/LEVEL1..4 → L1..L4`，其余直接 raise（仅重命名+别名归一，无转换规则） | **未映射 → 丢弃** |
| canonical（`data/canonical/<ds>/all.json`） | 原样保留（provenance）；另解析出 `target.category_id` 为唯一训练身份 | 原样保留（provenance；已验证 4 数据集 100% 保真） | 不存在 |
| SFT / RL parquet（`data/sft|rl/…`） | 标签 = `target.category_id`；messages 只含 prompt-visible 元数据（默认 field_name/field_description/field_type） | **不导出**（SFT row 与 RL five-field 均无 data_level） | 不存在 |

关键代码事实：
- `src/agent/task/contracts.py::SampleTarget` 仅含 `leaf_level/leaf_name/category_id/category_path`，不含 data_level。
- `src/agent/training/common.py::canonical_target`：训练标签唯一来自 `target.category_id`；`classification.level_1..4` 明确 "provenance only, never fallback"。
- 全仓 grep：`data_level` 在 `src/agent`、`script/verl`、`script/canonical` 中 0 命中；仅 预处理（processor/split/rft_export）、mapping、`script/analysis`（只读统计）触及。

## 3. 各数据集数据来源与已知事实

### finance（信托核心系统）
- 原始文件：`data/raw/部分金融数据.xlsx`（568 行 ↔ processed 568；按 (table,field) 566/566 data_level 一致）；标准文档 `data/raw/金融行业数据安全分类分级标准指南.xlsx`。
- 分类体系：标准指南 → `financial_standards_dict.json`（233 条）→ finance corpus 233 / registry（path ID）。
- data_level 来源：`部分金融数据.xlsx`"数据级别"列（原始 1/2/3/4）；与标准指南"最低安全级别参考"（1级~4级）**一致**：529/568 精确对齐，37 条可经同域相邻类对齐（如 单位联系人信息 L2 ↔ 单位联系信息 2级），仅 2 条离群（AMONEY、HXTRADENO）。
- data_level 分布：`L1:25 / L2:460 / L3:76 / L4:7`；L4 全部为"个人身份鉴别信息/传统鉴别信息"。
- field_sensitive：无。等级语义（1~4 级代表什么）：**仓库内无定义文本 → 不猜测**。

### shougang（首钢京唐集团）
- 原始文件：`data/raw/关基-数据分类分级目录.xlsx`（"首钢京唐数据分类分级目录"，234 叶，含**"分级"列** per-leaf `1/2/3`，分布 1:16/2:170/3:48）；测试子集导出 `data/raw/关基设施…用于测试(1).xlsx`。
- 分类体系：目录 → `guanji_dict.json`（234 条，**只保留类别、丢掉了"分级"列**）→ shougang corpus 233 / registry（code ID `A1-1-1`）。
- data_level 来源：目录"分级"列；shougang 已观测的 192 个 leaf category 中，sample `data_level` 与目录对应 category 的"分级"**192/192 一致、零冲突**（32 个目录类别在数据中未出现）。因此该目录是目前可确认的 `standard_data_level` 来源——但 sample 与 standard **概念上不预设必然相等**（见 §1 两个变体）。
- data_level 分布：`L1:899 / L2:14,188 / L3:4,328`；占位符 `——` 不可训练（1,022 条）。
- field_sensitive：主训练数据无；仅 64 行的测试导出文件带该列。等级语义：**无定义**（目录只有数字）。

### infra（钢铁基建，= shougang 测试子集）
- 原始文件：`data/raw/关基设施…用于测试(1).xlsx`（64 行）；逐条 (db,table,field) 与 infra processed **64/64 一致**（含 data_level）；⊂ shougang。data_level 的 `standard_data_level` 来源同样是该目录（经 shougang 复用），不另设独立映射。
- 分类/registry：同 shougang（registry_source=shougang）。
- data_level 分布：`L2:18 / L3:46`（原"分级"值 2/3）。
- field_sensitive：**有**（"字段是否敏感"，是=46 / 否=18，与分级不同步，见 §1）；预处理丢弃。
- 等级语义：无定义。

### pers_info（高校个人信息）
- 原始文件：`data/raw/带分级分类的个人基础信息样本190条.xlsx`（189 行 → 176，去重后；176/176 data_level 一致）。
- 分类体系：单层 `level_4` 18 类（放入 `level_4` 槽位，见 §1）；corpus 来自数据集自身 universe（`build_report.source = null`）。
- data_level 来源：该 Excel "分级"列直接 `L1/L2/L3`；**仓库无对应标准文档 → 仅有 sample label，无确认的 standard knowledge 来源**。
- data_level 分布：`L1:31 / L2:98 / L3:47`；1 类跨级（基本信息年级信息和班级信息 L1×1/L2×8）。
- field_sensitive：无。等级语义：仓库内 `education_dict.json` 为"公开/内部/重要/敏感"四档，与 L1~L3 无法对应且有反例（考核信息=内部数据却标 L3 等）→ 不猜测。

## 4. Phase 0 冻结的核心模型

```text
classification                 data_level
  sample label                 sample label
      ↓                            ↓
canonical category_id       normalized data_level
      │                            │
      └──────────┬─────────────────┘
                 ↓
      proposed Stage2 joint target
         category + data_level
      （Stage2 contract 待实现 / future target）


              原始分类分级标准
                         │
                         ↓
                canonical standard（Phase 1）
              category_id / description / path
              standard_data_level
              grading_rules（若能获得）
                         │
                         ↓
             为模型提供分级规则/参考
             + 与 sample data_level 审计


field_sensitive = 独立 field-level annotation，不属于上述两者
```

> **最关键的独立分级设计原则**：`standard_data_level` 是 category 的标准参考等级，**不预设其必然等于每个具体字段的最终 `sample_data_level`**；若目标是独立分级，Stage2 **不应直接把候选类别对应的 gold `standard_data_level` 暴露给模型**，否则 data_level 任务会退化为 category→level 查表。

- **Phase 0 冻结**：`data_level` 是什么（sample label 的分级标签）、来自哪里（finance/shougang/infra 可追溯到标准/目录；pers_info 尚无标准来源）、当前代码未训练它（provenance only）。
- **Phase 0 不决定**：`data_level` 最后训不训练；不把"当前未训练"写成"设计上不训练"。
- **设计方向（非当前实现，待 Phase 1 起逐步验证）**：从原始分类分级标准构建 canonical standard（含 `standard_data_level`），再与样本 `data_level` 做一致性审计——因此建 standard 时按各源真实分类深度建模，勿把 `level_4` 槽位当作真实标准第四层。

## 5. 目标任务方向与前置缺口

**目标任务方向：优先采用任务形态 B —— 模型独立判断字段安全等级。**

```text
Stage1:
field metadata + 全量 leaf categories
→ Top-5 categories

Stage2:
field metadata
+ Top-5 category descriptions/examples
+ 领域信息
+ 分级规则/标准说明
→ category + data_level
```

其中：
- `category` 与 `data_level` **都是模型预测目标**。
- `standard_data_level` 作为标准参考和审计字段保存。
- **默认不直接把每个候选的 `standard_data_level` 暴露给 Stage2**，否则 data_level 任务会退化为 category→level 查表。

**Baseline A（降级为基线）：lookup-assisted grading**
向模型提供候选 category 的 `standard_data_level`，用于评估"分类 + 标准读取"任务，并作为 independent grading（Target B）的对照基线。

**Target B：independent grading**
不直接给候选 gold level，模型依据字段语义、业务上下文和分级规则自行判级。

研究问题由此清晰为：
- **A：会不会选对标准项？**
- **B：会不会真正做分级判断？**

**其他未决项**：
- holdout 缺口：L4 仅 finance 7 条且全在 train；L1 在多数 val/test 缺失 → 分级泛化评估需处理（补充/重抽样/保持冻结）。
- `field_sensitive` 是否建设为字段级敏感标签，及其与 data_level 的关系定义。
- 跨数据集 L1~L4 **不推断同义**：finance / pers_info / shougang 无证据表明相同 `L3` 用同一套业务定义，仅 infra=shougang 同源明确。
- **研究风险（需专门实验验证）**：per-category `data_level` 近确定性映射（跨级样本全库仅 4 条）——若训练集里几乎每个 category 永远只有一个 level，即使目标是独立分级，模型仍可能学成**隐式 `category → level` 查表**而非真正分级规则。后续必须专门设计实验检查：构造同 category 多 level / 跨级样本；并至少跑三组对照——**A lookup-assisted grading vs B independent grading vs B−（对 grading rules 做消融）**——用于区分模型学到的是查表还是真正分级规则。
