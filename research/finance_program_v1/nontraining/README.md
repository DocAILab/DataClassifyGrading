# Program 动态引用与事实性恢复（2026-10-11）

这是用户批准的独立新版本，基于上一份115项冻结源码。历史实验及失败结果只读，不替换历史身份。本轮只加强显式动态参数引用约束，以及Program结束后的事实性恢复上下文。

Planner 优先使用 `arguments:{}`，由原binder补齐required参数；optional省略仍不注入。显式动态参数（canonical reference以`$state.`开头）必须是确切规范引用，不得手写对象、列表、null、数字或普通字符串。初始domain/field_name等保持原契约。无效计划在执行前拒绝，只计Planner尝试预算，保留原文和具体错误。

模型仍从全部 14 个可选 Skill 中自行选择步骤和顺序，每次 Planner 调用生成一个完整有序 Program。没有必选 Skill、固定完整流程示例、自动插入步骤或改写模型答案。可能生产者只表示静态可能性；元数据取得、输出发布、状态绑定以及最终终止，都必须由真实执行和冻结 guard 验证。`semantic_query`、`semantic_evidence` 与 `field_semantics` 保持各自契约含义，描述语义不能替代字段名语义。

核心入口文件：

- `program_loop/planning_contracts.py`：根据 catalog 和共享校验规则生成无状态值的规划说明。
- `program_loop/catalog_context.py`：把说明附加到每个 Skill 的原始契约库中，供 Planner 读取。
- `program_loop/planner.py`：说明参数省略、合法引用、依赖顺序和条件性执行的使用规则。
- `program_loop/validation.py`：静态依赖与显式动态引用约束，共享取值策略。
- `program_loop/recovery_context.py`：纯事实性恢复摘要，不执行Skill、不改state。
- `program_loop/controller.py`：Program级执行/反馈与全局预算循环；不在Skill间规划。

恢复消息从真实state与observation陈述实际成功/失败步骤、失败位置、尚未执行的尾部、证据失效与真实剩余预算。尾部是历史意图，不是强制路线；字段存在不等于已验证ready。最后一份Program也仍允许部分计划，代码不自动完成分类或赠送预算。合法终止仍由原terminal guard确认。

新增CPU测试只脚本化模型I/O，parser、binder、executor和guard用真实实现。`vendor/`、底层execution、Registry、P14池及`local_smoke.py`/`download_model.py`字节保持不变。完整变更与未改SHA由新source manifest记录，不以人工描述替代哈希证据。

用户已授权测试/审查门通过后在新远程目录单次运行同三条无标签输入、同9B基础模型、同生成参数和预算。无新adapter/训练、Val调策略、Test、依赖升级或Git提交推送。工程通过不等于模型成功；真实试跑结果及原始调用见交付后的`实验结果.md`与`SMOKE_README.md`，保留全部失败，不调prompt重复挑结果。

RED/GREEN、独立Spec/Quality审查、全CPU验收、完整diff、冻结新签名和证据ZIP在`reviews/`。两项修改同时施加，与历史三条只作描述性比较，不能分别归因或推广为准确率提升。DCG全文输入模式、池规模和训练方式不属本轮改动，不声称已实现导师全部行动指南。
