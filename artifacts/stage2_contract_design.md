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
- leaf 为占位符（shougang `——`）或缺失时 resolver 返回 `None`（由 pipeline 过滤），
  计数在 `resolver.skipped` / `resolver.code_fallbacks` 中暴露。
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

## 3. 每个 dataset 的 category_id 策略

| dataset | 策略 | category_id 形态 | 依据（阶段 1 报告） |
|---|---|---|---|
| shougang | `code` | guanji code，如 `A1-1-1`、`B1-2` | 192/193 leaf 与 guanji 精确对齐；233 code 全部唯一；字母↔L1 100% 一致（A=研发/B=生产/C=管理）。**数字段是不透明序号，不解释为 level_2/3**。占位符 `——`（1022 样本）→ 无 target |
| infra | `code`，`registry_source="shougang"` | 同 shougang（4/4 leaf 同 code） | infra ⊂ shougang；不构建 4 类别的独立 registry（LeafRegistry 有 ≥5 条校验） |
| finance | `path_hash` | `finance:<sha256(4段path)[:16]>`，如 `finance:3bf4a5fc…` | fs 是 3 段 path、dataset 是 4 级，**不把 join(level_1~4) 当作 corpus identity**；hash 只保证稳定+无碰撞。3 个真同名不同父 leaf（单位基本情况/基本信息/交易清结算信息）各得 2 个 ID；"配置信息"的空白变体（经营 管理/经营管理）归一化后同 1 个 ID |
| pers_info | `path_hash` | `pers_info:<sha256(leaf)[:16]>` | 仅 level_4（18 唯一，单级）；corpus 只覆盖 4/18，schema 允许 target 存在而 description 缺失 |

统一 fallback：`code` 策略下 leaf 无 code（corpus 不完整）→ 自动回退 path_hash 并计数
（`resolver.code_fallbacks`），不报错、不修标签。

## 4. 新增 / 修改文件

新增：
- `src/agent/task/dataset_config.py` — `DatasetConfig` + `BUILTIN_DATASET_CONFIGS`（四数据集策略）
- `src/agent/task/identity.py` — `stable_category_id` / `code_leaf_map` / `build_leaf_registry` / `compact`
- `src/agent/task/resolver.py` — `TargetResolver`（Protocol）/ `ClassificationTargetResolver` / `resolve_all`
- `tests/task/test_targets.py` — 19 个 contract 测试（覆盖 6 项要求）
- `cfg/task/dataset.example.json`、`cfg/task/corpus.example.json` — schema 示例

修改：
- `src/agent/task/contracts.py` — `LeafCategory` 增加 `name`（缺省回退 category_id，旧 registry JSON 兼容）/
  `path` / `code`；新增 `CorpusCategory`、`SampleTarget`
- `src/agent/task/__init__.py` — exports

未修改（刻意不动）：`processor.py`、corpus builder、SFT exporter、prompts、VeRL 数据格式。

## 5. 留到阶段 3 的问题（故意不在此解决）

1. **corpus → registry 正式接入**：finance 的 fs 3 段 path ↔ 数据集 4 级 path 的映射、
   `corpus:finance` bare-leaf 文档如何挂到 category 上、description/examples 填充。
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
