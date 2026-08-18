# 阶段 3C：canonical data contract 接入 SFT 数据流水线

## 1. 修改/新增文件

修改：
- `src/agent/training/sft/dataset.py` — 标签唯一来源改为 `target.category_id`（canonical contract）；删除 classification level_4 读取（无 fallback）；新增 canonical + split-by-id join 导出；resolved-only 过滤；target↔registry 一致性校验
- `src/agent/task/prompts.py` — Stage 1 渲染完整 registry（category_id + name，不放 description）；Stage 2 通过 category_id 从 canonical corpus 获取 category_id/name/description/descriptions/examples
- `script/verl/sft/export.py` / `validate.py` — CLI 改为 `--canonical` + `--split-dir` + `--corpus`
- `docs/SFT_BASELINE.md` — CLI 用法同步
- `tests/sft/`（fixtures canonical 化 + 签名更新）

新增：
- `script/verl/sft/prompt_stats.py` — prompt 长度统计（字符级 + 可选 tokenizer）
- `tests/sft/test_sft_canonical_e2e.py` — 真实数据端到端（CI 无 data/ 时 skip）
- `tests/sft/test_prompt_stats.py`、`tests/sft/fixtures/canonical/`
- `artifacts/prompt_stats_{finance,shougang}.json`、`artifacts/prompt_token_stats_{finance,shougang}.json`

## 2. 新 SFT 数据流

```
data/<ds>/canonical/all.json  (resolution_status + target)
  → resolved-only loader      (status == resolved, target 存在,
                                category_id ∈ registry, leaf_name 匹配)
  → 按原 split id 边界 join data/<ds>/{train,val,test}.json
  → Stage1/Stage2 examples    (Stage1: 完整 registry catalog; Stage2: corpus bundle)
  → parquet                   (data/sft/<ds>/)
  → VeRL-compatible dataset
```

## 3. Stage 1 / Stage 2 输入输出 schema

- Stage 1 user：`[{"category_id": "...", "name": "..."}, ...]`（**完整** registry，无 description/examples）；assistant：`{"candidates": ["<category_id>" ×5]}`
- Stage 2 user：candidate bundle（category_id/name/description/descriptions/examples，由 category_id 查 canonical corpus）；assistant：`{"answer": "<category_id>"}`
- ground truth = `target.category_id`（两阶段一致）；classification 保留为 provenance

## 4. 四 dataset 实际可训练 resolved 数量（真实 pipeline）

| dataset | canonical resolved | 进入训练集（split 边界内） | 备注 |
|---|---|---|---|
| finance | 531 | **529** | 2 条 resolved 记录不在原始 split 边界内（数据管线事实，未静默补入） |
| infra | 64 | **64** | |
| pers_info | 176 | **176** | |
| shougang | 18,393 | **18,393** | |

skipped（不进训练）：finance 37（34 missing_leaf + 3 path_mismatch）、shougang 1,022（`——` placeholder）等，全部与 canonical report 一致。

## 5. registry / target consistency

- 每条进入训练的 sample：`target.category_id in registry.ids` 且 `target.leaf_name == registry.get(category_id).name`（不匹配即 fail-fast 报错，无静默跳过）
- Stage 1 catalog 条目数 == registry size（e2e 断言）
- Stage 2 bundle 全部经 category_id 查 corpus（缺则报错）
- 无 classification["level_4"] fallback（专门测试：level_4 与 target 不一致时 ground truth 取 target）

## 6. parquet 统计与 VeRL compatibility

- `data/sft/<ds>/{train,val,test}.parquet`：finance 806/138/114 rows、infra 80/30/18、pers_info 280/36/36、shougang 29042/3852/3892（= resolved × 2 stages）
- pyarrow + pandas 读取正常；columns: messages/stage/source_id/metadata/ground_truth/candidates
- 行数 == resolved × 2；无 unresolved sample；两次导出字节级 deterministic
- `validate_sft_dataset` 全 split valid；verl-compat 测试（MultiTurnSFTDataset 加载）签名已更新，CI verl 环境执行

## 7. Stage 1 prompt token 长度（Qwen2.5 tokenizer，chat template）

| dataset | stage1 tokens (max / p95) | stage2 tokens (max) | 32,768 上限 |
|---|---|---|---|
| finance | 5,959 / 5,950 | 794 | 未超 |
| shougang | 5,174 / 5,076 | 436 | 未超 |

（本地 0.5B 与 7B 同属 Qwen2.5 家族共享 tokenizer；7B 实测留待服务器。Stage 1 因全量 registry catalog 占 ~5-6K tokens，**是后续优化的主要对象**，本阶段只检测报告。）

## 8. 本地已完成验证

89 passed, 1 skipped（含真实数据 e2e 8 项：resolved 数、registry 一致、universe 完整、corpus 唯一获取、validate 通过、deterministic）。prompt 长度统计（字符 + tokenizer）。pandas/pyarrow 读取。

## 9. 仍需服务器执行的 smoke test

- Qwen2.5-7B-Instruct tokenizer 实测 prompt 长度（预计 ≈ 0.5B 结果）
- VeRL MultiTurnSFTDataset 加载真实 parquet（CI verl job 覆盖）
- 真实 trainer step（阶段外）

## 10. 留给下一阶段 RL 接口的问题

- Stage 1 全量 registry catalog 是 prompt 长度大头（~5-6K tokens）：RL 阶段若引入检索/压缩需先改 retrieval policy（CandidatePolicy 未实现，当前 fixture 明确非生产）
- finance 2 条 split 边界外 resolved 记录：数据侧需决定是否补回 split
- `data_level` 敏感级别语义仍未确认（RL reward 若使用需先确认）
- 37 条 finance unresolved 的 SFT 覆盖决策（已从训练集排除）
