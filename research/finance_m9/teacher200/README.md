# 金融字段分类分级 · Teacher200 SFT 与 Planner GRPO

同一份 Val422 上，学生 Planner 经过 SFT 后类别和等级同时答对 **198/422**，加上过程奖励进行 RL 后答对 **200/422**。从 SFT 到 RL，6 个字段由错转对、4 个由对转错。两阶段共用 Qwen3.5-9B、冻结的 Scale1169 分类 Skill、同表查证工具和评估字段。

| Val422 指标 | 学生 SFT | RL（1 / 0.12 / 0.02） |
| --- | ---: | ---: |
| 类别和等级均正确 | 198/422（46.9%） | 200/422（47.4%） |
| 等级正确 | 240/422（56.9%） | 241/422（57.1%） |
| 合法终止 | 419/422（99.3%） | 418/422（99.1%） |
| 使用最常见的完整工具顺序 | 345/422（81.8%） | 350/422（82.9%） |

[SFT 逐字段结果](results/sft/逐字段结果.csv) · [RL 逐字段结果](results/rl/Val422逐字段.csv) · [SFT 详细指标](results/sft/评分.json) · [RL 详细指标](results/rl/评分.json)

## 数据和代码如何衔接

1. **教师与 SFT 数据。** 百炼教师 `qwen3.8-2.4t-a95b` 在 TRAIN 字段上采集轨迹。200 个入选字段都有通过查证且同时命中类别和等级的结果。导出器对照实际 Planner 请求、回复和成功执行的程序，保留 201 个监督回合，跳过 111 个被拒绝或执行失败的回合。教师采集时核对了学生可见信息。部署到 M9 后，201 条训练提示与部署提示全文相同的有 0 条，这项格式差异记录在独立核对表中。[采集覆盖](teacher/README.md) · [采集器](../../../finance_teacher/runtime/collector/small_test.py) · [导出逻辑](sft/export.py) · [提示核对](sft/input-audit.json)
2. **学生 SFT。** 在 Qwen3.5-9B 上训练 Planner LoRA，监督程序 JSON 和结束符。201 条示范按原顺序训练一轮，以 16 为批大小，共 13 次优化器更新。每次更新保存权重、优化器和随机状态。分类 Skill 始终使用已训练好的 Scale1169。[训练代码](../train_sft.py) · [断点恢复](../teacher_checkpoint.py) · [学生权重](weights/sft) · [冻结分类器](weights/skill) · [重新加载核验](sft/reload-check.json)
3. **Planner RL。** 从上述 SFT 权重初始化策略和冻结的参考策略。先对 32 个训练字段试采，再用 128 个字段做 16 批在线 GRPO，每批 8 个字段、每字段 4 条轨迹，共 512 条。模型在采样时只接收已取得的证据。最终合法且类别、等级均正确得 1 分；正确类别进入分类前最后一次有效检索的前五名得 `0.12 / 排名`；字段描述和表描述的引用均通过核验得 0.02 分。弃权或未通过终止校验得 0 分。只更新 Planner。第 10 批起支持多卡分担回合梯度，每批由一个进程执行一次优化器更新。[固定字段与种子](rl/plan.json) · [奖励实现](rl/skill_reward.py) · [训练入口](rl/campaign.py) · [多卡梯度合并](rl/parallel_replay.py)
4. **选模型与评估。** 第 4、8、16 批各评估 41 个开发字段，分别答对 18、17、15 个。取分数最高的第 4 批，在全部 422 个字段上贪心评估。逐字段记录预测、工具顺序、停止原因和调用次数。RL 评分一共核验了 1,226 条采样与评估记录。[选中的第 4 批](results/rl/updates/batch-04) · [训练终点第 16 批](results/rl/updates/batch-16) · [评分实现](rl/score.py)

## 离线复核

进入本 README 所在目录。下面的命令使用 Python 3.10 或更高版本。代码仓库包含教师记录、SFT 原始评估轨迹、逐字段结果和推理用的适配器。RL 原始轨迹、全部 16 个 RL 断点和全部 13 个 SFT 断点放在同一版本的 GitHub Release 中。[下载附件](https://github.com/DocAILab/DataClassifyGrading/releases/tag/finance-teacher200-sft-grpo-20261127) · [文件大小和 SHA-256](release-assets.json)

```bash
python sft/export.py
python sft/score_full.py
python fetch_artifacts.py
python rl/score.py
```

第一步在 `sft/planner.jsonl` 生成实际训练所用的字节，SHA-256 应为 `a222faaba464b0a5800782f0b6a6bd930befeb8eda1698b7d193f52f208f019e`。下载程序校验附件，再恢复到评分脚本使用的目录。已手动下载附件时可加 `--from-dir /path/to/downloads`，另加 `--sft-checkpoints` 可恢复 SFT 的 13 个断点。两套评分均按来源 ID 与参考答案对齐，以全部 422 条为分母。

## 在 GPU 上重跑

使用 Linux、支持 BF16 的 NVIDIA GPU 和本地 Qwen3.5-9B 基座。基座 revision 为 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。运行时会核对模型文件。实际实验环境见[版本清单](environment.json)，参数和目录可从以下命令直接复用。

```bash
export FINANCE_BASE_MODEL=/path/to/Qwen3.5-9B
export CUDA_VISIBLE_DEVICES=0
python sft/export.py
python sft/gpu_queue.py
export FINANCE_SFT_ADAPTER="$PWD/work/sft/planner-adapter"
python sft/full_eval.py queue
python rl/campaign.py queue
```

已有随仓库发布的 SFT 权重时，直接设置 `FINANCE_BASE_MODEL`、`CUDA_VISIBLE_DEVICES` 并执行最后两行即可。已发布 RL 实验固定 SFT 与分类器的 SHA-256；复现原始 RL 可用 `weights/sft`。重新训练得到的 SFT 若哈希不同，需要为新实验建立对应的训练计划。可用 `FINANCE_RL_RUN` 指定新的 RL 输出目录。训练中断后，以相同的目录和参数重新执行队列命令即可续跑。分配多卡时将 `CUDA_VISIBLE_DEVICES` 设为允许使用的卡号列表。

SFT 评估与 RL 队列会逐条保存预测，并在完成后归档和评分。新产生的结果默认写入 `work/sft-eval` 和 `work/rl`。已有完整运行目录时，可执行 `python archive_run.py rl /path/to/completed-run` 重新归档并评分。需要重新请求百炼时，设置自己的 `DASHSCOPE_BASE_URL` 和 `DASHSCOPE_API_KEY`，然后执行 `python teacher/recollect.py /path/to/new-output 1095`；模型调用将产生新的费用。
