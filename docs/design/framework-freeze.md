# Framework Freeze（阶段 13.5）

本文件把两阶段叶分类工程的关键接口声明为 **frozen**。冻结后，后续算法实验
（GRPO / RLOO / ReMax / PPO-DAPO 等）除真正 bug 外**不得修改**这些层；
算法差异只允许落在 launcher / Hydra 配置 / 算法层（见 §完整流程清单）。

冻结日期：Phase 13.5（2026-08-19，Phase 11/12/13 smoke 与 7B SFT baseline 均已 PASS）。

## 1. Frozen contracts（冻结接口清单）

| # | 层 | 内容 / 入口 | 不得修改的点 |
|---|---|---|---|
| 1 | canonical category_id | `agent.task.contracts.LeafRegistry` / `SampleTarget`；`data/<ds>/canonical/all.json` | 唯一身份：`target.category_id`；`classification.level_*` 仅 provenance，禁止 reintroduce level_4 fallback |
| 2 | PromptChoice protocol | `agent.task.prompt_choices.PromptChoiceRegistry`（global "1".."N" / stage2 local "1".."5"） | choice id 映射、display name 规则；canonical id 永不进 prompt/不作为模型动作 |
| 3 | candidate policy | `agent.task.prompts.build_stage1_prompt`（全 registry）/ `build_stage2_prompt`（5 candidate bundle，按 category_id 查 corpus） | 现行为准：Stage1 全目录无 description；Stage2 bundle=5；hard-negative 策略不抽象（沿用 GT+registry 前四 fixture 策略） |
| 4 | parquet schema | SFT：`messages/stage/source_id/metadata/ground_truth/candidates`；RL：`data_source/prompt/ability/reward_model/extra_info` | 列名与语义；label 唯一来源 `target.category_id`；assistant supervision = choice-id JSON |
| 5 | parser / decode | `agent.task.parser.check_stage1_choices / check_stage2_choices / check_stage{1,2}_output` | 严格 JSON + 精确 schema + 无 fuzzy fallback；decode 到 canonical 后才进入正确性逻辑 |
| 6 | reward contract | `agent.training.rl.reward`（RewardConfig 0/0.3/1.0 + stage2_partial(0.5 provisional)） | 数值表与 `reward_stage{1,2}_choices` / `reward_for_choice_result`（decode 先行，表不变） |
| 7 | VeRL reward adapter | `agent.training.rl.verl_adapter.compute_score` | choice-aware 路由（`<dataset>/stage<1|2>`），不实现 parser/reward 表 |
| 8 | evaluation protocol | `agent.evaluation.classification`（`evaluate_stage{1,2}_choices`）+ `script/verl/sft/evaluate_baseline.py`（factorized/proxy）+ `script/verl/sft/evaluate_true_e2e.py`（true E2E） | 指标定义：stage1 format/contract/Recall@5；stage2 conditional / format/contract；true E2E（Stage2 用 Stage1 预测 Top-5） |

## 2. 模型起点与 RL 初始化（冻结）

- 所有正式实验统一使用 **Qwen/Qwen2.5-7B-Instruct**（小规模功能链路可用
  Qwen2.5-0.5B-Instruct 作为 config 开关打通，不改变语义）。
- RL 共同初始化默认 **SFT final = `global_step_140`（merged HF `merged_step140`）**
  （Phase 13）；该选择仅作起点，后续若完成更完整的 SFT 训练，再统一重选一个
  SFT checkpoint，选定后同样冻结。
- **不得用 test 指标选择 checkpoint**（Phase 13.5 起为硬约束）。

## 3. 完整流程清单（数据处理 → 训练 → 评估 → 其他 RL 算法）

```text
学长预处理/corpus
   └─> data/<ds>/canonical/all.json + split JSON（split 按 id 隔离）
        ├─> script/verl/sft/export      （canonical → SFT messages parquet，choice protocol）
        ├─> script/verl/rl/export       （canonical → RL 五字段 parquet）
        └─> validate / check_token_budget  （契约 + token 预算 gate）

SFT 训练：script/verl/sft/run_baseline.sh → verl.trainer.sft_trainer（LoRA/FSDP/bf16）
   └─> checkpoints（global_step_N，verl LoRA FSDP，含全量 base 权重 ~15GB/个 → 必须配 SAVE_FREQ+MAX_CKPT_KEEP）
        └─> script/verl/sft/merge_lora_checkpoint → 合并 HF 目录（可给 eval / RL 加载）

评估（统一 greedy / prompt / parser / seed=42）：
   ├─> script/verl/sft/evaluate_baseline.py     — factorized/proxy 指标（Stage2 用 parquet 预构造 bundle）
   └─> script/verl/sft/evaluate_true_e2e.py     — true E2E（Stage2 由 Stage1 预测 Top-5 动态构建）

RL（在冻结层之上只改算法层）：
   script/verl/rl/grpo_smoke.sh -> verl.trainer.main_ppo
     ├─ reward.custom_reward_function.path=pkg://agent.training.rl.verl_adapter（冻结）
     └─ algorithm.* / actor.* / rollout.*（launcher 层，可换）
```

**实现其他 RL 算法（RLOO / ReMax / PPO-DAPO）的改动面**（各 4B/12 报告约定）：
- 只改 `script/verl/rl/grpo_smoke.sh` 的算法/采样配置：`algorithm.adv_estimator`
  （grpo→dapo/rloo），RLHF-style 需加 critic/ref/`use_kl_loss` 相关配置。
- rollout 后端 / 显存旋钮在 launcher env（`ENFORCE_EAGER` / `PARAM_OFFLOAD` /
  `GPU_MEM_UTIL` / `MAX_MODEL_LEN` 等）。
- 若必须新增 reward 形态：只在 `agent.training.rl.verl_adapter`（薄路由）后加
  compute_score 分支，parser/reward 表不动。
- **不需要改**：canonical/parquet/parser/prompt/reward 表/evaluator/VeRL 源码
  （零 vendored patch，兼容只靠 pip 依赖 + 配置）。

## 4. True-E2E evaluator（阶段 13.5 新增）

`script/verl/sft/evaluate_true_e2e.py`：raw test → `build_stage1_prompt` →
greedy → `check_stage1_choices` decode 真实 canonical Top-5 → 按这 5 个
category_id 查 corpus → `build_stage2_prompt` 动态构建 → greedy →
`check_stage2_choices` local decode → final canonical → 与 GT 比较。
输出：stage1 format/contract/Recall@5；stage2 conditional acc（仅 GT∈预测
Top-5）+ format/contract 失败率；true E2E acc。metric 命名：
**Proxy E2E**（`evaluate_baseline` 的 `proxy_e2e`）与 **True E2E**（本 evaluator）必须区分。
单测：`tests/evaluation/test_evaluate_true_e2e.py`（stage1 miss / hit+correct /
hit+wrong / malformed 不 crash / candidates==预测 Top-5）。
