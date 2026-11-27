# 金融分类分级训练主线（M9）

一个字段进来时，规划器先看到字段名，再决定要查哪些元数据。检索模块从 12 个金融类别中选出 5 个候选，分类器根据已经查到的证据选择类别并给出引用。控制器检查引用和终止条件，按选中标准条目给出等级。证据不足时，流程可以弃权。

同一个 Qwen3.5-9B 基座挂两套 LoRA 权重。Teacher200 学生负责规划，Scale1169 负责分类。GRPO 只更新规划权重。

M9 和新采集的教师共用[查表工具](../finance_teacher/runtime/collector/dsl_table_tools.py)。在单一逻辑数据库内，已取得的表名可以定位同表其他字段。查字段名只返回名称，查询选中的字段或完整表上下文后才能取得相应描述。表名缺失时无同表结果。若多个已知数据库出现同名表，工具用数据库名消除歧义。

复核实验可直接使用仓库内的 [Val422 输入、参考标签、逐字段评分和 nontraining 运行配置](val422/README.md)。其中也提供了 422 个字段的实际首计划，方便核对固定流程对照的设计。

## 当前训练与评估入口

[Teacher200 SFT → Planner GRPO](teacher200/README.md) 给出本次主线的代码、字段划分、环境、训练命令、权重与 Val422 报告。Planner 用 200 个教师字段导出的 201 条示范训练，分类 Skill 使用冻结的 Scale1169。RL 采用结果、检索、引用三项奖励，权重分别为 1、0.12、0.02。

| 模块 | 实现 |
| --- | --- |
| Planner 和分类 Skill 的 LoRA 训练 | [train_sft.py](train_sft.py)、[断点](teacher_checkpoint.py) |
| 原生执行、同表查询和终止检查 | [episode.py](episode.py)、[运行时组装](native_assets.py) |
| 冻结基座检索、Scale1169 分类、Planner 路由 | [model.py](model.py)、[routing.py](routing.py) |
| 本次训练队列与奖励 | [campaign.py](teacher200/rl/campaign.py)、[skill_reward.py](teacher200/rl/skill_reward.py) |
| Val422 评分 | [SFT](teacher200/sft/score_full.py)、[RL](teacher200/rl/score.py) |

独立的 [nontraining 入口](../finance_program_v1/nontraining/README.md) 和 [Val422 数据说明](val422/README.md) 可用于复核既有实验。
