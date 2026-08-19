# Classification Framework Freeze（阶段 13.5）

本文件把**分类（classification）管道**的关键接口声明为 **frozen**（classification
pipeline / classification contract **frozen for experiments**）。冻结范围**只覆盖
classification**（§1 清单）；**data_level / grading 不在冻结范围内**（见 §1.1，语义
待确认）。冻结后，后续算法实验（GRPO / RLOO / ReMax / PPO-DAPO 等）除真正 bug 外
**不得修改**这些层；算法差异只允许落在 launcher / Hydra 配置 / 算法层（见
§完整流程清单）。

冻结日期：Phase 13.5（2026-08-19，Phase 11/12/13 smoke 与 7B SFT baseline 均已 PASS）。

## 1. Frozen contracts（分类管道冻结接口清单）

| # | 层 | 内容 / 入口 | 不得修改的点 |
|---|---|---|---|
| 1 | canonical category_id | `agent.task.contracts.LeafRegistry` / `SampleTarget`；`data/<ds>/canonical/all.json` | 唯一身份：`target.category_id`；`classification.level_*` 仅 provenance，禁止 reintroduce level_4 fallback |
| 2 | PromptChoice protocol | `agent.task.prompt_choices.PromptChoiceRegistry`（global "1".."N" / stage2 local "1".."5"） | choice id 映射、display name 规则；canonical id 永不进 prompt/不作为模型动作 |
| 3 | candidate / Stage 输入输出契约 | `agent.task.prompts.build_stage1_prompt` / `build_stage2_prompt` | 冻结的是**契约**：Stage1 恰好输出 Top-5 candidate；decode 后为 canonical category_ids；Stage2 恰好接收 5 个 predicted candidate category_ids；Stage2 候选顺序决定 local choice id 1..5。**不冻结**：离线合成候选构造（build_candidates 为 fixture 而非生产检索策略）、负采样 / hard-negative 策略、候选采样改进（见 §1.2） |
| 4 | parquet schema | SFT：`messages/stage/source_id/metadata/ground_truth/candidates`；RL：`data_source/prompt/ability/reward_model/extra_info` | 列名与语义；label 唯一来源 `target.category_id`；assistant supervision = choice-id JSON |
| 5 | parser / decode | `agent.task.parser.check_stage1_choices / check_stage2_choices / check_stage{1,2}_output` | 严格 JSON + 精确 schema + 无 fuzzy fallback；decode 到 canonical 后才进入正确性逻辑 |
| 6 | reward contract（框架） | `agent.training.rl.reward` + `reward_for_choice_result` | 冻结：choice 输出→严格 parser→choice decode→canonical category_id→reward adapter 路由→malformed/constraint-invalid 处理框架。**不冻结**：仍标 provisional 的 reward 系数（如 stage2_partial=0.5 数值）与后续由任务定义确认的 reward weighting（见 §1.3）；当前 reward.py 数值不动 |
| 7 | VeRL reward adapter | `agent.training.rl.verl_adapter.compute_score` | choice-aware 路由（`<dataset>/stage<1|2>`），不实现 parser/reward 表 |
| 8 | evaluation protocol | `agent.evaluation.classification`（`evaluate_stage{1,2}_choices`）+ `script/verl/sft/evaluate_baseline.py`（factorized/proxy）+ `script/verl/sft/evaluate_true_e2e.py`（true E2E） | 指标定义：stage1 format/contract/Recall@5；stage2 conditional / format/contract；true E2E（Stage2 用 Stage1 预测 Top-5） |

### 1.1 Not frozen / pending clarification（data_level / grading）

以下内容**当前不冻结**，仓库内语义 / 契约未定：

- `data_level` 语义（L1–L4 业务定义 / 分级规则 / 标准文档）——仓库无正式定义，UNKNOWN
- grading target schema（分级目标的数据结构）
- grading prompt / output contract（分级提示词与输出格式）
- grading parser / evaluator（分级解析与评估）
- grading reward（分级奖励）
- classification + grading 联合指标（分类与分级联合评估）

> `data_level` is currently preserved as source provenance only. Its promotion to a
> supervised grading target is pending confirmation of the L1-L4 business
> definition and task protocol.

未来若加入 grading，是在 **frozen classification contract 之上新增一个 grading
contract**，而不是重新设计 classification pipeline；classification 层保持冻结不动。

**grading 允许 additive 扩展**：若后续确认 Stage2 需同时输出 classification + grading，
允许对 Stage2 output schema 做 **additive** 扩展，例如
`{"answer":"3","data_level":"L2"}`。冻结的是「`answer` choice → canonical category_id」
映射与分类正确性语义，**不是**「Stage2 JSON 只能有 answer 一个字段」。classification
identity / PromptChoice 分类映射不变；grading 契约后续作为新增契约引入。

### 1.2 Not frozen：synthetic candidate construction（fixture，非生产策略）

当前 SFT/RL 数据流用 `build_candidates()`（GT + registry 前四个 negatives +
deterministic permutation）为样本合成候选；源码明确它属 **fixture，不是生产 Stage1
检索策略**，因此**不冻结**，`build_candidates()` 本身也不改：

- 离线 SFT/RL 合成候选构造
- 负采样策略 / hard-negative 策略
- 未来候选采样 / 召回改进（正式检索需按需求另行设计）

候选相关冻结的只有 §1 表 row 3 的输入输出**契约**。

### 1.3 Not frozen：provisional reward coefficient

- 冻结：reward **框架**（choice 输出→严格 parser→canonical decode→reward adapter
  路由→malformed/constraint-invalid 的 0/partial 映射框架）。
- **不冻结**：仍标 provisional 的 reward 数值（例如 Stage-2 `stage2_partial` 当前默认
  0.5）；最终权重待任务定义确认后由配置决定。当前 `reward.py` 数值不动。

## 2. 模型起点与 RL 初始化（分类实验约束）

- 所有正式实验统一使用 **Qwen/Qwen2.5-7B-Instruct**（小规模功能链路可用
  Qwen2.5-0.5B-Instruct 作为 config 开关打通，不改变语义）。
- Phase 13 的 SFT checkpoint（如 `global_step_140`）是 reproducible SFT baseline /
  candidate downstream-initialization artifact，**不冻结**任何具体 checkpoint；RL
  初始化选择属于实验配置（run config），不在本冻结契约内。
- **不得用 test 指标选择 checkpoint**（Phase 13.5 起为硬约束）。

## 3. 完整流程清单（数据处理 → 训练 → 评估 → 其他 RL 算法）

```text
预处理/corpus
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

RL（在冻结的分类层之上只改算法层）：
   script/verl/rl/grpo_smoke.sh -> verl.trainer.main_ppo
     ├─ reward.custom_reward_function.path=pkg://agent.training.rl.verl_adapter（分类路径，frozen）
     └─ algorithm.* / actor.* / rollout.*（launcher 层，可换）
```

> 候选构造（`build_candidates`：GT+registry 前四 negatives+固定 permutation）目前是
> **合成 fixture**，非生产检索策略；正式候选采样 / 召回需按需求另行设计（见 §1.2）。

### 3.1 各阶段输入/输出示例（真实数据 · pers_info）

**(1) 输入 · 数据处理前（raw canonical record）**

> **待分类数据是什么**：一个数据库**字段（column）**。原始记录 metadata 存全量字段信息
> （库/表名、字段名、字段描述、类型、值），经 `TaskConfig.metadata_fields` 显式裁剪后
> 才进入 prompt（见 (3)，pers_info/finance/infra 均只暴露 `field_name, field_description`）。

```json
{
  "id": "f374612b-7a1b-52c4-97b0-fb0851603dd6",
  "key": "dpname",
  "label_status": "labeled",
  "metadata": { "database_name": "CCENSE", "table_name": "M_BASE_CUSTDEPT",
                 "field_name": "DPNAME", "field_description": "部门" },
  "classification": { "level_1": "", "level_4": "学校概况基本信息" },
  "resolution_status": "resolved",
  "target": { "leaf_level": "level_4", "leaf_name": "学校概况基本信息",
               "category_id": "pers_info:学校概况基本信息" }   // 唯一 label 来源
}
```

**(2) 处理后 · SFT parquet 行（`data/sft/pers_info/test.parquet`, stage1）**

```text
stage=stage1   source_id=016e8ce6-…   ground_truth=pers_info:教职工个人基本信息
messages[0] system  = You are a leaf-category candidate retriever. …（见 (3)）
messages[1] user    = （见 (3) 的确切 user prompt）
messages[2] assistant (gold supervision, choice-id JSON) = {"candidates":["2","11","4","3","1"]}
```

**(3) Stage 1 prompt + output**

system（逐字）：
```
You are a leaf-category candidate retriever. Return exactly one JSON object with key "candidates". The value must contain exactly five unique choice ids from the catalog. Do not output Markdown, commentary, canonical category ids, or any other keys.
```
user（逐字，目录节选：完整 18 项为 `[choice_id, display_name]`，无 description/无 canonical id）：
```json
Retrieve five candidate leaf categories from this catalog:
[["1", "人力资源数据"], ["2", "任课信息"], ["3", "基本信息年级信息和班级信息"], …(共 18 项)…, ["18", "课程信息"]]
Field metadata:
{"field_name":"eid","field_description":"指导老师工号"}
```

> **待分类数据（在 input 中的位置）＝ user prompt 末尾 `Field metadata` 块**：
> 它就是上面那条 **字段**。原始记录 metadata 为
> `{"database_name":"USR_DATAI","table_name":"T_JW_BKSCXCY","field_name":"eid",
>  "field_description":"指导老师工号","field_type":"STRING", …}`，
> 经 `TaskConfig.metadata_fields` 只暴露前两个（其余如库/表名、字段类型、值都不进 prompt）。
> 模型的任务=**仅凭这个字段（名+描述）从目录里选出 5 个候选叶类**。
模型输出（SFT final 实际, hit case）→ choice decode → canonical Top-5：
```json
{"candidates":["4","1","2","3","11"]}
  "4"  → pers_info:学历学位信息        "1"  → pers_info:人力资源数据
  "2"  → pers_info:任课信息            "3"  → pers_info:基本信息年级信息和班级信息
 "11"  → pers_info:教职工个人基本信息   ← GT ∈ Top-5 ✓
```
malformed 输出（不 crash，判 format/contract fail → reward 0）：
```json
{"candidates":[{"id":17,"name":"职称信息"}, …]}   // 旧 id/name 对象形状，非 choice-id 字符串数组
```

**(4) Stage 2 prompt（由 Stage1 预测 Top-5 动态构建）+ output**

system（逐字）：
```
You are a leaf-category reranker. Return exactly one JSON object with key "answer". Its value must be one of the five candidate ids "1" through "5". Do not output Markdown, commentary, or any other keys.
```
user（逐字 — bundle 的 5 个 candidate **= 上面预测的 Top-5**，local id 1..5，description/examples 取自 canonical corpus）：
```json
Candidate bundle:
[{"id":"1","name":"学历学位信息","description":"","descriptions":[],"examples":[]},{"id":"2","name":"人力资源数据",…},{"id":"3","name":"任课信息",…},{"id":"4","name":"基本信息年级信息和班级信息",…},{"id":"5","name":"教职工个人基本信息",…}]
Field metadata:
{"field_name":"eid","field_description":"指导老师工号"}
```
输出 → local bundle-id decode：
```json
correct   : {"answer":"5"}  → 预测 Top-5[4] = pers_info:教职工个人基本信息 == GT → e2e TRUE
wrong     : {"answer":"1"}  → 预测 Top-5[0] = 学历学位信息            != GT → e2e FALSE
malformed : "1"（裸 id，base 常见）→ format fail → 0，不 crash
```

**(5) 处理后 · RL parquet 行（`data/rl/pers_info/train.parquet`，五字段，brief）**

```text
data_source  = pers_info/stage1 | pers_info/stage2
prompt       = [system, user]（与 SFT 相同 choice-protocol 文本，无 assistant gold）
ability      = data_classification
reward_model = {"ground_truth": "pers_info:…", "style": "rule"}
extra_info   = {"candidates": null | [5 个 canonical id], "dataset": "pers_info",
                "metadata": {"field_name":…, "field_description":…}, "source_id": …}
```

**实现其他 RL 算法的改动面**：
- 只改 `script/verl/rl/grpo_smoke.sh` 的算法/采样配置：`algorithm.adv_estimator`
  （grpo→dapo/rloo），RLHF-style 需加 critic/ref/`use_kl_loss` 相关配置。
- rollout 后端 / 显存旋钮在 launcher env（`ENFORCE_EAGER` / `PARAM_OFFLOAD` /
  `GPU_MEM_UTIL` / `MAX_MODEL_LEN` 等）。
- 若必须新增 reward 形态：只在 `agent.training.rl.verl_adapter`（薄路由）后加
  compute_score 分支，parser/reward 表不动。
- **不需要改**：canonical/parquet/parser/prompt/reward 表/evaluator/VeRL 源码
  （零 vendored patch，兼容只靠 pip 依赖 + 配置）。

## 4. True-E2E evaluator

`script/verl/sft/evaluate_true_e2e.py`：raw test → `build_stage1_prompt` →
greedy → `check_stage1_choices` decode 真实 canonical Top-5 → 按这 5 个
category_id 查 corpus → `build_stage2_prompt` 动态构建 → greedy →
`check_stage2_choices` local decode → final canonical → 与 GT 比较。
输出：stage1 format/contract/Recall@5；stage2 conditional acc（仅 GT∈预测
Top-5）+ format/contract 失败率；true E2E acc。metric 命名：
**Proxy E2E**（`evaluate_baseline` 的 `proxy_e2e`）与 **True E2E**（本 evaluator）必须区分。
单测：`tests/evaluation/test_evaluate_true_e2e.py`（stage1 miss / hit+correct /
hit+wrong / malformed 不 crash / candidates==预测 Top-5）。

### 4.1 true-E2E 样例（SFT final · pers_info test · greedy/seed=42）

**【success】source=fb23c995-…   gt=pers_info:教职工个人基本信息**
```text
stage1 输出 : {"candidates":["4","1","2","3","11"]}
            → decode Top-5: [学历学位信息, 人力资源数据, 任课信息, 基本信息年级信息和班级信息, 教职工个人基本信息]  ← 含 GT ✓
动态 stage2 : 用该 Top-5 构建 bundle（见 §3.1(4)，bundle id = 该 Top-5 的位置）
stage2 输出 : {"answer":"5"}  → 断言 stage2 candidates == 预测 Top-5 ✓
final       : pers_info:教职工个人基本信息 == GT → TRUE E2E ✓
```

**【fail：Stage1 miss】source=016e8ce6-…   gt=pers_info:教职工个人基本信息**
```text
stage1 输出 : {"candidates":["3","4","2","1","17"]}
            → decode Top-5: [基本信息年级信息和班级信息, 学历学位信息, 任课信息, 人力资源数据, 职称信息]  ← 不含 GT ✗
stage2 输出 : {"answer":"4"}  → 人力资源数据（预测 bundle 第 4 项）
final       : pers_info:人力资源数据 ≠ GT （GT 不在候选，Stage2 无法命中）→ TRUE E2E ✗
```

**【fail：malformed stage2】（stage1 命中但 stage2 非 JSON）**
```text
stage1 输出含 GT，stage2 输出裸 "1"（base 常见）→ format/contract fail → reward 0 → TRUE E2E ✗（不 crash，结构化失败）
```
