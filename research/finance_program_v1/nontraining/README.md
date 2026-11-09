# 未训练 Planner 条件（NONT-D / ProgSkill）

金融主干的 non-training 是 `evaluation/evaluate.py` 中的 `base` arm：原始 Qwen3.5-9B Planner，加同一 F66 Skill 库。`sft` arm 仅给 Planner 开启第一版 LoRA，Skill 仍关闭 adapter。因此两臂共享 `../evaluation/runtime/`，不再保留第二套 P14 控制器副本。

从 [实验协议](../docs/EXPERIMENT_PROTOCOL.md) 和 [运行说明](../REPRODUCE.md) 阅读条件与入口。历史 P14 的 runner、下载器和源码审计不属于422配对结果，已从当前金融主干移除；原代码及文件SHA可从 [历史记录](../docs/HISTORY.md) 追溯。

本次仅归纳原版逻辑，没有合入本地 recovery 的失败记忆、防重复或新预算补丁。原版仍可能出现预算耗尽、反馈覆盖和不合规模型输出，详见协议中的限制。
