# 阶段 2：最小统一数据 contract 设计

> 依据：`artifacts/data_alignment_report.md/.json`（阶段 1）。
> 原则：不自动修复报告中的 UNKNOWN 数据问题；`classification.level_1~level_4`
> 原样保留作为 provenance；训练 ground truth 最终只依赖 `target.category_id`。

## 1. 最终 canonical sample schema

```jsonc
{
  // 原始 record（不变，provenance）：
  "classification": { "level_1": "...", "level_2": "...", "level_3": "...", "level_4": "..." },
  "label_status": "labeled",
  "metadata": { "...": "..." },
  "data_level": "L2",            // 仅 provenance；不解释为 classification depth

  // resolver 派生（训练消费）：
  "target": {
    "leaf_level": "level_4",     // 该 dataset 的 canonical leaf level
    "leaf_name": "合同归并",      // 展示名；不保证唯一
    "category_id": "B1-2",       // 训练 GT 唯一依赖；dataset LeafRegistry 内唯一
    "category_path": ["生产数据域", "生产合同（订单）", "合同归并"]  // 非空段；provenance，不是 ID
  }
}
```

- `target` 是**派生结构**，由 `ClassificationTargetResolver` 从 record 生成；不写回 classification。
- 解析结果显式三分：
  - **structural skip**：无 classification / leaf 为空 / 占位符（shougang `——`）→ 无 target，计 `skipped`；
  - **unresolved**（仅 code 策略）：leaf 存在但 corpus 无 code → 无 target，在
    `resolver.unresolved` 按 leaf 名显式报告；**不静默 fallback 到其他 ID 方案**；
  - resolved：生成 `SampleTarget`。
- 需要"corpus 不完整但仍产生 target"的数据集（pers_info）使用自己的
  deterministic path 策略，而不是依赖 fallback。
- `category_id` 只在一个 dataset 的 LeafRegistry 内唯一（跨 dataset 允许相同，
  例如 infra 与 shougang 共享同一 code 空间）。

## 2. 最终 canonical corpus schema

```jsonc
{
  "categories": [
    {
      "category_id": "A1-1-1",          // 唯一；与 LeafRegistry 对齐
      "name": "科研设备预约管理",        // leaf 名；不假设唯一
      "description": "指科研检验设备预约过程中产生的信息…",  // 可空（pers_info 无 corpus 也合法）
      "path": ["研发数据域", "科研设备管理", "科研设备预约管理"],  // 可空；根到 leaf 名称链
      "code": "A1-1-1",                 // 可空；不透明稳定标识，数字段不做 level 解释
      "examples": ["设备预约信息", "审核表", "取还设备记录"]   // 一个 category 多个 example；可空
    }
  ]
}
```

允许：一个 category 多个 description/example；category 没有 description；corpus 不完整
（无全局完整性校验）。不再使用 bare level_4 name 作为通用 category identity。

**Registry 与 corpus 的关系**：生产用 `LeafRegistry` 代表 classification standard /
corpus 的**完整 leaf universe**，由 `leaf_registry_from_corpus(categories)` 构建，
**绝不从训练样本 SampleTarget 推导**（训练样本可能只覆盖 universe 的子集）。

## 3. 每个 dataset 的 category_id 策略

| dataset | 策略 | category_id 形态 | 依据（阶段 1 报告） |
|---|---|---|---|
| shougang | `code` | guanji code，如 `A1-1-1`、`B1-2` | 192/193 leaf 与 guanji 精确对齐；233 code 全部唯一；字母↔L1 100% 一致（A=研发/B=生产/C=管理）。**数字段是不透明序号，不解释为 level_2/3**。占位符 `——`（1022 样本）→ structural skip；corpus 中 leaf 无 code → unresolved（显式报告，无 fallback） |
| infra | `code`，`registry_source="shougang"` | 同 shougang（4/4 leaf 同 code） | infra ⊂ shougang；不构建 4 类别的独立 registry（LeafRegistry 有 ≥5 条校验） |
| finance | `path` | `finance:<L1>.<L2>.<L3>.<L4>`，如 `finance:业务.账户信息..基本信息` | fs 是 3 段 path、dataset 是 4 级，**不把 join(level_1~4) 当作 corpus identity**。ID 是 deterministic、human-readable、path-qualified（非 opaque hash）：固定 4 槽、空 level_3 保留空槽（`..`），同名不同父必然不同 ID。3 个真同名不同父 leaf（单位基本情况/基本信息/交易清结算信息）各得 2 个 ID；"配置信息"的空白变体（经营 管理/经营管理）归一化后同 1 个 ID |
| pers_info | `path`（path_fields=level_4） | `pers_info:<leaf>`，如 `pers_info:学籍管理信息` | 仅 level_4（18 唯一，单级）；corpus 只覆盖 4/18，schema 允许 target 存在而 description 缺失 |

category_id 与 display name（`LeafCategory.name` / `SampleTarget.leaf_name`）保持分离：
ID 是身份，name 是展示；name 不参与 identity 判定。

## 4. 新增 / 修改文件

新增：
- `src/agent/task/dataset_config.py` — `DatasetConfig` + `BUILTIN_DATASET_CONFIGS`（四数据集策略）
- `src/agent/task/identity.py` — `qualified_category_id` / `code_leaf_map` / `leaf_registry_from_corpus` / `compact`
- `src/agent/task/resolver.py` — `TargetResolver`（Protocol）/ `ClassificationTargetResolver` / `resolve_all`
- `tests/task/test_targets.py` — 21 个 contract 测试（覆盖 6 项要求 + unresolved 语义）
- `cfg/task/dataset.example.json`、`cfg/task/corpus.example.json` — schema 示例

修改：
- `src/agent/task/contracts.py` — `LeafCategory` 增加 `name`（缺省回退 category_id，旧 registry JSON 兼容）/
  `path` / `code`；新增 `CorpusCategory`、`SampleTarget`
- `src/agent/task/__init__.py` — exports

未修改（刻意不动）：`processor.py`、corpus builder、SFT exporter、prompts、VeRL 数据格式。

## 5. 留到阶段 3 的问题（故意不在此解决）

1. **corpus → registry 正式接入**：finance 的 fs 3 段 path ↔ 数据集 4 级 path 的映射、
   `corpus:finance` bare-leaf 文档如何挂到 category 上、description/examples 填充；
   各 dataset 的 canonical corpus JSON（`CorpusCategory` 列表）落地。
2. **候选召回**：leaf registry 冻结后，SFT fixture（GT+顺序前 4）替换为 corpus 驱动的
   hard-negative 策略（`CandidatePolicy`）。
3. **shougang `——` 占位符策略**：过滤（当前行为）vs 单独类别，需数据侧决策。
4. **infra 与 shougang 的 registry 复用**：确认训练时 infra 直接用 shougang registry
   （当前设计），或阶段 3 改为合并导出。
5. **LeafRegistry ≥5 条硬校验**：SFT fixture 时代的约束；若阶段 3 出现 <5 类的 dataset
   需重审该校验（infra 通过 registry_source 绕开）。

## 6. 仍需人工确认的 UNKNOWN（不阻塞本阶段）

- finance 5 个 leaf 无任何语料定义（`交易清金额信息`疑似错别字、`基本信息（公开`截断、
  `单位基本信息/单位基本情况/单位联系人信息`命名不一致）→ 阶段 3 前需定名。
- `data_level` 字段语义（与分类深度矛盾）——若阶段 3 用作敏感级别，需确认映射。
- guanji code 数字段 ↔ level_2/3 的对应（需原指南；本阶段按不透明 ID 处理）。
- pers_info 14/18 leaf 的语料来源未知。
- education_dict code（A1-1 等）与 pers_info 无层级可对照。
- shougang corpus 中 1 条 malformed 条目（`"nan"`，JSON NaN）— 无 code 且不在数据集
  leaf 内，阶段 3 清理语料时需处理。
