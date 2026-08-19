# 阶段 4A：统一 RL 数据、解析与 reward contract

状态：完成（未实现训练算法、未跑服务器训练）。对应 milestone M2→M4 之间的数据/契约前置。

## 1. 修改 / 新增文件

修改：

- `src/agent/task/parser.py` — 新增统一 parser 层 `ParseResult` + `check_stage1_output` / `check_stage2_output`（不抛异常），保留原 raising parser 不变
- `src/agent/evaluation/classification.py` — `evaluate_stage1/2` 改为消费统一 parser（公开行为与返回类型不变，单点实现 format/constraint 判定）
- `src/agent/task/__init__.py` — 导出 `ParseResult` / `check_stage1_output` / `check_stage2_output`
- `src/agent/training/sft/dataset.py` — 抽出共享 helper 到 `agent.training.common`（调用点不变，无行为改动）
- `docs/RL_FRAMEWORK.md` — RL 落地状态与训练算法 RL 代码的准入条件更新
- `.github/workflows/ci.yml` — verl-compat job 增加 `verl.utils.dataset.rl_dataset` 导入 + `tests/rl/test_rl_verl_compat.py`
- `AGENTS.md` — 项目结构同步（以下未逐条展开）

新增：

- `src/agent/training/common.py` — `canonical_target` / `build_candidates` / `require_corpus` / `require_corpus_covers_registry`（SFT 与 RL 共享）
- `src/agent/training/rl/__init__.py` / `sample.py` / `reward.py` / `dataset.py` — RL 数据 / reward / VeRL parquet 契约
- `script/verl/rl/__init__.py` / `export.py` / `validate.py` — CLI 适配器
- `tests/rl/test_rl_parser.py` / `test_rl_reward.py` / `test_rl_dataset.py` / `test_rl_canonical_e2e.py` / `test_rl_verl_compat.py`
- `docs/phase_reports/stage4a_report.md`（本文件）

## 2. RL sample schema

`agent.training.rl.sample.RlSample`（冻结 dataclass）：

| 字段 | 类型 | 说明 |
|---|---|---|
| stage | str | `"stage1"` / `"stage2"` |
| dataset | str | 数据集名（进 `data_source`） |
| source_id | str | canonical 记录稳定 id |
| messages | tuple[RlMessage×2] | system + user（**无 assistant gold**） |
| ground_truth | str | 唯一来源 `target.category_id` |
| candidates | tuple[str×5] \| None | 仅 Stage 2（fixture 策略，非正式 retrieval） |
| metadata | Mapping[str,str] | task config 显式 metadata 字段 |
| reward | RewardMeta | reward 路由元数据（dataset + stage） |

构建规则：仅 `resolution_status == "resolved"` 进入；建立 canonical index 时，**所有 resolved 记录都调用共享的 `canonical_target()` 验证 target contract**（无论是否属于 train/val/test split）：target 存在、category_id 非空且 ∈ LeafRegistry、leaf_name 与 registry 一致、resolved 记录有稳定 id。split 只决定是否进入训练，不决定是否执行 contract validation。ground truth 为 `target.category_id`（与 registry 强校验）；**禁止 classification["level_4"] fallback**。不包含任何训练算法内部状态。

## 3. VeRL RL parquet schema（verl v0.8.0 实测）

对照 verl v0.8.0 `RLHFDataset`（`prompt_key` 默认 `"prompt"`，纯文本 2 条 message；
reward manager 读取 `non_tensor_batch["reward_model"]["ground_truth"]`）实现，非旧版格式：

| column | 类型 | 内容 |
|---|---|---|
| data_source | string | `<dataset>/stage1` 或 `<dataset>/stage2`（reward 路由） |
| prompt | list<{role,content}> | [system, user]；rollout 由 VeRL 追加 generation prompt |
| ability | string | `task_config.task_name`（默认 `data_classification`） |
| reward_model | struct | `{"style": "rule", "ground_truth": "<category_id>"}` |
| extra_info | struct | `{dataset, stage, source_id, metadata, candidates?(stage2)}` |

约束：resolved-only；建立 index 时全部 resolved 记录 target contract 必须验证通过、`ground_truth ∈ registry`、确定性（两次导出字节相同）、pyarrow 可读、validator 可重建两阶段 prompt 校验、不包含训练算法内部字段、不包含 SFT assistant gold response、**每个 source_id 恰好 1 条 stage1 + 1 条 stage2（重复 stage 行 → validator failure）**。

## 4. Parser contract（统一 parser，`agent.task.parser`）

`check_stage1_output(text, *, registry, expected_count=5)` / `check_stage2_output(text, *, candidates)` → `ParseResult`

`ParseResult{format_valid, constraint_valid, output, errors}`，`ok = format_valid and constraint_valid`；**对模型输出永不抛异常**。

- Stage 1 校验：JSON 合法 → schema 正确（仅 `candidates`）→ 数量==5 → 无重复 → 全部 ∈ registry
- Stage 2 校验：JSON 合法 → schema 正确（仅 `answer`）→ answer ∈ candidates（candidates 数量==5 、无重复、全 ∈ registry）
- **Stage 2 不要求 `ground_truth ∈ candidates`**：明确的两阶段任务语义 — Stage 1 可能召回失败，因此 `ground_truth ∉ candidates` 是合法状态，不属于数据 contract 错误，也不会被当作 Stage 2 schema 错误
- candidates 自身非法（非 5 唯一 / 空）视为编程错误抛 `ValueError`（其来源是 dataset 而非模型）

`evaluate_stage1/2` 与 reward 均消费同一层，判定逻辑收敛为单点。

## 5. Reward contract（`agent.training.rl.reward`）

`RewardConfig{stage1_valid_miss=0.3, stage2_partial=0.5}`；`RewardResult{reward, reason, parsed_output, format_valid, task_correct}`

| 档位 | Stage 1 | Stage 2 |
|---|---|---|
| 格式/schema/约束非法 | 0.0 | 0.0 |
| 合法但任务未成 | 0.3（`stage1_valid_miss`） | `stage2_partial`（**默认 0.5，待确认，配置项**） |
| 正确 | 1.0 | 1.0 |

- Stage 1 `task_correct` = ground truth ∈ candidates；Stage 2 = answer == ground truth
- parser 错误不会抛训练中断异常（永不 raise）
- reward 只描述任务正确性，不含任何训练算法逻辑
- `ground_truth ∉ candidates`（Stage 1 召回失败）时，Stage 2 输入仍合法、parser 正常、合法 candidate answer 可正常计算 reward，不抛异常

**`stage2_partial` 是 provisional / pending task-policy confirmation 的配置项**：默认 0.5 仅作当前过渡值，不当作已确定的最终 reward policy，本阶段不调优具体数值。

## 6. 四数据集真实 export + validate 结果（本地真实 data/，2026-08-18）

| dataset | canonical resolved | trainable resolved | rows(train/val/test) | skipped_not_resolved | validate |
|---|---|---|---|---|---|
| finance | 531 | 529（2 条在 split 边界外） | 806 / 138 / 114 | 36 / 0 / 1 | ✅ valid |
| infra | 64 | 64 | 80 / 30 / 18 | 0 / 0 / 0 | ✅ valid |
| pers_info | 176 | 176 | 280 / 36 / 36 | 0 / 0 / 0 | ✅ valid |
| shougang | 18,393 | 18,393 | 29,042 / 3,852 / 3,892 | 826 / 102 / 94 | ✅ valid |

rows == trainable_resolved × 2（stage1+stage2）。全部 `validate_rl_dataset(valid=True)`；两次导出字节级 deterministic。

Stage 1 prompt 真实 tokenizer 边界（只记录不改）：finance ~5959 / infra ~5079 / shougang ~5174 / pers_info ~523。

## 7. tests / CI

- 本地全量（无 verl 标记 2 个文件）：**145 passed**（含既有 sft/eval/task 回归）
- `tests/rl/`：parser（invalid JSON / wrong schema / duplicate / unknown / wrong count / valid-wrong / correct）、reward（各档位 + config 覆盖 + 永不抛异常 + 编程错误仍 raise + RewardResult 字段）、dataset（unresolved 不进入、**resolved 且不在任何 split 但 target.category_id ∉ registry → export fail-fast**、target.category_id 唯一 label、classification 不参与 GT、**Stage2 不要求 GT∈candidates 的保护测试**、**duplicate/missing stage 行 → validator failure**、Stage1 全 registry、Stage2 corpus 按 category_id、列=VeRL 五字段、无 assistant、deterministic、validate 接受/报错、全 unresolved split fail-fast）、四 dataset e2e、verl-compat（`RLHFDataset` 加载，CI verl job 执行）
- `script/verl/rl/{export,validate}.py` CLI 已本地 smoke（pers_info 176 → train 280 / val 36 / test 36，validate valid）

## 8. 尚未确认的 task policy

1. **Stage 2 partial reward 具体值**：`RewardConfig.stage2_partial` 默认 0.5，首条真实 RL vertical slice 需确认
2. **正式 Stage 1 retrieval policy**：当前 candidate fixture（GT + registry 前四）是一个
   temporary deterministic fixture，不是正式 production retrieval policy；本阶段不绑定任何
   具体未来 retrieval 实现，正式策略在后续阶段另行确定。
3. **Stage 1 max_length / 截断策略**：finance Stage1 ~5959 tokens（truncation=error 下正式训练 max_length ≥ 6200 或先定 Stage 1 截断策略）——本阶段只记录

## 9. 下一阶段训练接口需要消费的字段

训练侧（GRPO/PPO vertical slice）消费：

- `data_source`：reward 路由（`<dataset>/stage1|stage2`）
- `prompt`：system+user messages → RLHFDataset tokenize + rollout generation prompt
- `reward_model["ground_truth"]`：唯一 label（`target.category_id`）
- `extra_info["candidates"]`（stage2）：Stage 2 reward 的答案集合；`extra_info["stage"]` 区分档位；`extra_info["source_id"]` 日志/审计
- `reward_stage1` / `reward_stage2`（或 `reward_for_parse_result`）：作为 VeRL reward 的薄适配底层，reward 表单点实现

VeRL reward adapter 形态：`compute_score(data_source, solution_str, ground_truth, extra_info, ...)` → 按 `data_source` 路由到 `reward_stage1` / `reward_stage2`，`extra_info` 提供 stage/candidates。
