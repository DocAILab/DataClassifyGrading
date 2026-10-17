# non-training 与第一版 Planner SFT 实验汇报

日期：2026-10-17。发布范围仅为 non-training、teacher 轨迹与第一版学生 SFT；本分支不包含后续训练实验。

## 已完成的工作

1. 建立真实 Program 规划、Skill 执行、事实性反馈和严格终止的 non-training 基线。
2. 从真实 teacher 执行轨迹中筛选并复核 200 个唯一训练来源，另 14 个备用不训练。
3. 用 Qwen3.5-9B 进行 Planner-only LoRA SFT，完成 1 epoch、13 次更新。
4. 在固定开发 Val80 上完成原基座 Planner 与 SFT Planner 的同接口配对评估，完整回放后本地事后计分。
5. 公开代码、CPU 测试及 440 条脱敏教师/学生轨迹视图，原始业务数据和完整证据保留在受控环境。

## 1. 历史 non-training 基线

2026-10-11 的 P14 Program 基线不使用 adapter、不训练。模型每次生成完整有序 Program，从 14 个 Skill 中自主选择，执行结束或失败后得到事实性反馈，允许预算内重规划。无固定插入步骤或自动补答案。

| 指标（完整 80 来源分母） | 历史 non-training P14 |
|---|---:|
| 合法终止 | 69/80（86.25%） |
| 类别正确 | 21/80（26.25%） |
| 等级正确 | 38/80（47.50%） |
| 类别与等级联合正确 | 21/80（26.25%） |
| 首次／最终／任次 Top5 命中 | 49/80（61.25%） |
| 8 类／固定 12 类 MacroF1 | 0.317683／0.211789 |
| 模型调用 | 603 |

11 条未终止及其他错误保留，不过滤分母。smoke2 不计正式 80。该历史设置使用 P14、单 EOS 等旧接口，**不能把 26.25% 与下面 SFT 的 45% 当成严格同接口提升**；严格对照是下一节重新执行的原基座 vs SFT。

## 2. Teacher 与第一版 SFT

Teacher 是百炼 `qwen3.8-2.4t-a95b`（Qwen3.8-2.4T），Planner／模型型 Skill 真实调用，确定性 Skill 真实执行。经过格式、终止、类别等级标签核对及无明显重大推理问题的基本复核，选出 200 个唯一来源；这不是专家语义真值认证。这批轨迹用于学生的 SFT 冷启动，并非训练或微调 teacher 本身，也不是由 9B 学生自产的监督数据。

学生监督是实际 Planner Program JSON + EOS。原输入两消息保留，prompt 全 mask；不监督 Skill 回复、原始 reasoning 或事后标签答案，teacher 生成输入不放 gold。Val/Test 不作为训练生成或采样依据。

| 项目 | 配置／实测 |
|---|---|
| 学生基座 | Qwen3.5-9B，固定 revision |
| LoRA | r=8、alpha=16、dropout=0.05 |
| 学习率 | 1e-4 |
| 微批／梯度累积 | 1／16，末 8 样本保留 |
| 数据 | 200 唯一训练来源，14 备用不训练 |
| Epoch／实际更新 | 1／13 |
| 最长真实训练输入 | 22,428 token，完整不截断 |
| 正式训练墙钟 | 9,975.78 秒，约 2 小时 46 分 |
| 训练 loss | 0.147918，仅训练目标值 |

最长真实输入试运行通过后，从 fresh 原基座正式训练，试运行权重不复用。训练与归档真实 exit0；CPU 独立核对全来源/token/mask/EOS/更新数/基座 SHA/有限非零 LoRA，实际 40 语言目标层、80 个 A/B 张量，无视觉参数更新。

## 3. 第一版 SFT 的同接口固定 Val80 对照

两组仅 Planner 参数不同：原基座 vs 第一版 SFT LoRA。模型型 Skill 均本地原基座、禁用 Planner adapter，确定性 Skill 真实执行。F66 全库、controller、预算、greedy、双 EOS 和上下文限制一致，每来源交替组先后顺序。

每组固定 80，共 160 正式轨迹；另 4 smoke 不计正式分母。全部失败、未终止保留分母。服务器完整回放、本地完整 controller 回放通过，gold 只本地事后计分。

| 指标 | 同接口原基座 Planner | 第一版 SFT Planner | 变化 |
|---|---:|---:|---:|
| 合法终止 | 36/80（45%） | 80/80（100%） | +55 个百分点 |
| 类别正确 | 16/80（20%） | 36/80（45%） | +25 个百分点 |
| 等级正确 | 26/80（32.5%） | 47/80（58.75%） | +26.25 个百分点 |
| 联合正确 | 16/80（20%） | 36/80（45%） | +25 个百分点 |
| 首次／最终／任次 Top5 命中 | 33/80 | 47/80 | +14 来源 |
| 8 类 MacroF1 | 0.196429 | 0.413290 | +0.216861 |
| 固定 12 类 MacroF1 | 0.130952 | 0.275527 | +0.144574 |
| 模型调用数 | 552 | 320 | 约 -42.0% |
| 输入 token | 5,629,680 | 2,082,528 | 约 -63.0% |
| 输出 token | 59,551 | 17,570 | 约 -70.5% |

联合正确逐来源配对：21 个提升、1 个退步，净增 20；合法终止 44 个提升、0 个退步。不能把训练 loss 当分类效果，也不能将该开发集结果称为独立 Test 泛化或显著性保证。Val80 只覆盖 12 类中的 8 类、单 seed，同表相关性和稀缺类别支持限制仍在。

## 4. 代码与轨迹交付

- nontraining/：历史 P14 controller、Skill 合同、真实 runner、审计与 CPU 测试。
- sft/：实际 teacher 监督样本提取、编码、completion-only loss、LoRA trainer、资源及阶段门。
- teacher/：百炼 qwen3.8-2.4t-a95b 的实际 SSE 后端、生成协议、有限重试与安全账本代码；不含凭证或真实部署清单。
- evaluation/：同接口原基座/SFT 路由、真实 evaluator、完整回放、结果计分。
- trajectories/teacher200.redacted.jsonl：200 个训练 teacher 的真实规划与结果投影。
- trajectories/nontraining_val80.redacted.jsonl：历史 non-training 全 80 来源。
- trajectories/student_val80_base.redacted.jsonl 与 student_val80_sft.redacted.jsonl：第一版对照评估两组各 80 来源。

公开轨迹替换原 source ID，移除业务元数据、messages、raw reasoning、Skill 文本观察及证据值；保留真实 Program/步骤、实际执行状态、最终结果和原 trace SHA。200 teacher 及 80 SFT 学生 Program 参数均无需脱敏，保持原值；4 条原基座轨迹存在显式字面参数，公开版明确替换为 [REDACTED]，不假装逐字完整。所有失败及未终止轨迹均保留。

这些公开视图是**有损审阅材料**，不是原始训练数据、不能完整 controller replay，也不据公开版重新声称原始输入语义可复核。原 SHA 供受控证据对照，不可从有损 JSON 重算原 trace SHA。模型权重、原始数据、gold、实验室地址/UUID、个人绝对路径、凭证不上传。

代码是隔离审阅快照，配置/数据需在获授权环境中重新冻结，克隆分支不会自动启动 API/GPU 任务。详细复现边界见 REPRODUCE.md，汇总指标及发布校验见 results/。

