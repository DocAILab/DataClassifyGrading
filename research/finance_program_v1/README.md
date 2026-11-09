# 金融 Program 主干：原版 F66 + Planner SFT

本目录围绕一个实验问题组织：**同一 F66 Program/Skill 框架下，只训练 Planner，是否能改善未训练 Planner 的分类与终止表现？** 主干是 Teacher200 → Planner-only LoRA SFT → `base` / `sft` 固定来源配对评估。

本次整理基于 GitHub 原版 `b6fde9b80802593ad8ab30bdfae66fc28483a849`，只整理金融 Program。运行代码的改动限排版和注释；提示词、Skill body、原控制流、预算、adapter 路由、训练目标和已有结果保持原版语义。2026-11-09 本地 recovery 修复版不在此目录。

## 先读哪些文件

1. [完整实验协议](docs/EXPERIMENT_PROTOCOL.md)：条件、数据、训练、推理、评分和可比范围。
2. [代码职责与数据流](docs/ARCHITECTURE.md)：从 Planner 到执行器、模型路由和审计。
3. [运行与验证说明](REPRODUCE.md)：可直接运行的 CPU 检查，以及真实复现还需要的资产。
4. [历史快照说明](docs/HISTORY.md)：为什么旧 P14 不再作为第二套 non-training 主干。

| 目录 | 主干职责 |
| --- | --- |
| `teacher/` | 百炼 `qwen3.8-2.4t-a95b` 的采集、显式策略、传输重试和账本 |
| `sft/` | 真实 Planner Program JSON + EOS 的数据导出、分词和第一版 LoRA SFT |
| `evaluation/runtime/` | **唯一主干 runtime**：原版 F66 控制器、binder、Skill 与 terminal guard；两臂共享 |
| `evaluation/` | 仅 Planner 启用 adapter 的路由、执行、原调用回放、身份检查和指标 |
| `nontraining/README.md` | 未训练 Planner 条件的导航；直接指向同一 F66 runtime，没有第二套复制代码 |
| `results/`、`trajectories/` | 原聚合结果和1044条脱敏轨迹，字节不变 |
| `tools/`、`publication_tests/` | 脱敏与公开依赖验证；CPU 检查不运行真实模型 |

## 已发布的同协议 Val422 结果

每组固定422来源，合844条正式轨迹。`base` 是**未训练 Planner 调用 Skills**，不是直接回答的 Base-LLM；`sft` 只启用第一版 Planner adapter，模型型 Skill 两臂均用基座。

| 指标 | base / ProgSkill | sft / DCG-SFT |
| --- | ---: | ---: |
| 类别匹配 | 79/422（18.72%） | 171/422（40.52%） |
| 等级匹配 | 111/422（26.30%） | 230/422（54.50%） |
| 合法终止 | 191/422（45.26%） | 420/422（99.53%） |
| 实际模型调用 | 2897 | 1695 |

逐来源有99改善、7退化。完整分母、Top5、F1、成本与阶段墙钟保存在 [evidence-summary.json](results/evidence-summary.json)。本次未重新训练、生成或评分；旧结果不是本次排版后的新实验。

Val422 是已观察开发集，只有8/12类有支持、来源可能共享业务上下文；它不提供专家 Gold 或独立业务族保证。合法终止不等于业务正确。公开轨迹脱敏有损，不能用于全量控制器回放或直接训练。M8/M9/M10、后续 Skill SFT 和其他 RL/领域实验不属于这份第一版 Planner 主干协议，不能与本表直接拼成统一对照。
