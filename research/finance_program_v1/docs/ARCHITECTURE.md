# 主干代码导读

```text
Teacher受控采集 -> 原始接受的Planner消息/Program -> export_dataset/token_audit
                                                        |
                                               Planner-only SFT adapter
                                                        |
同一来源 + F66 runtime -> base Planner / SFT Planner -> Skill(base) -> trace/calls/route
                                                        |
                                       exact replay / identity gates -> 本地metrics
```

## 外层规划与执行

`evaluation/evaluate.py` 负责固定来源、AB/BA和每臂输出目录；`backend.py`只负责实际模型与adapter路由。`io_utils.py`在一个明确边界初始化本阶段runtime/legacy导入。不同阶段的短模块名不要在同一Python进程混用。

`runtime/program_loop/controller.py`只协调 Planner→parse→执行Program→收集observation→终止/再次规划，并保存budget与trace。`planner.py`/`catalog_context.py`/`planning_contracts.py`生成公共契约及消息。`validation.py`检查Program结构、引用、依赖与长度；执行时 `execution.py`重新绑定实际state并交给底层guard。静态可能生产者不是已取得的事实。

`runtime/vendor/pool_runtime.py`、`pool_catalog.py`组合F66池与类型检查；底层 `vendor/runtime/code/agent/orchestration/model_skill_executor.py`组装Skill实际请求、读取模型输出并验证。`vendor/runtime/skills/full_pool/`是各Skill公共body；名称解释、描述、比较、分类与最终终止各有契约。名称/描述已使用grounded对象的路由不能称为独立名称推理。

## 训练与证据

`sft/export_dataset.py`保留来源与实际响应边界，`token_audit.py`做完整prompt前缀与Program+EOS监督，`completion_loss.py`核因果loss边界。`train_student.py`做最长样本trial与fresh formal；`pipeline.py`管资源锁、实际子进程退出与阶段门。

`evaluation/replay.py`消耗已记录模型回应，拒绝请求消息或调用顺序变化；`gate.py`核adapter/身份/真实退出；`runtime_identity.py`绑定运行源码；`metrics.py`保留固定分母。`evaluation/legacy/`仍是这些原版模块的必要模型/哈希依赖，不是第二个主干runtime，不能为目录看起来短而删除。

`teacher/teacher_api.py`处理SSE，`retry_backend.py`只重试授权范围内的传输问题，`campaign_budget.py`等记录安全账本；`policy.py`是显式版本化消息变体，`detached.py`是历史受控协调器。`pilot_audit.py`在采集回放中是必要依赖，虽然名字含pilot，也不应仅按名字删除。

源码排版展开了原单行多语句、条件和字典；注释集中解释来源隔离、预算、路由、监督和失败恢复。没有通过注释暗示原版不存在的修复。公共库仍支持部分旧接口，主干入口明确选F66。
