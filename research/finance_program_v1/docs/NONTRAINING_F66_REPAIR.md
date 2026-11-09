# F66 non-training 运行时修复与验证

本目录以原版金融 Program 主干为底座，只整理正式 F66 `evaluation/runtime` 的程序修复。原始版本来自 GitHub `b6fde9b80802593ad8ab30bdfae66fc28483a849`；修复依据为隔离运行版 `nontraining-f66-recovery-20261109-01/recovery03-cpu/runtime` 的冻结文件，冻结 SHA 为 `7484c93c4156e0048e97731733a03e10f4eb1f5e7ec22fed03c48b6dc8761f74`。旧版轨迹、失败与实验分母未被覆盖。

## 代码职责

- `evaluation/runtime/program_loop/controller.py` 控制 Program、Planner 与 Skill 预算，执行后汇总真实反馈，并在无新证据或无进展时停止。
- `failure_history.py` 保存同一来源内跨 Program 的失败事实；`input_readiness.py` 表示已获取、缺失和未知的输入状态；`progress.py` 区分新事实与模型措辞变化。
- `execution.py` 在执行前核动态输入与剩余预算。`grounded_request_guard.py` 只对字段名、字段描述、字段缩写三个已有 grounded Skill 按实际完整请求去重，底层已有返回后才登记；新事实或不同纠错消息仍可触发新请求。
- `provider_protocol.py` 统一公开 Skill 输出形状；`model_skill_executor.py` 按该协议向模型发出请求。Skill 文档与协议代码同步修改。显式不足保留为不足，不强行补成类别或业务事实。

## 回归与真实 smoke

合并到审阅副本后，在 `evaluation/runtime` 目录运行 `python -B -m pytest tests -q -p no:cacheprovider`：30 项通过；在 `evaluation` 目录运行同命令：25 项通过。新增测试使用明确合成的模型回复，检查跨轮失败、预算、重复请求、公开协议和合法终止，不能当作业务准确率。恢复版原目录中另有 61 项 CPU 测试及真实原生 IPC 控制；审阅副本只纳入与 `evaluation/runtime` 直接相关的三个测试文件。

为便于审阅，PR 中的 14 个 Python 改动文件又按 Black 25.9.0、88 字符行宽整理了格式。整理前后逐文件 Python AST 相同，两个目录的 30 项和 25 项 CPU 测试重新通过。**这一步只改变审阅副本的排版；PR 中文件字节与原先真实 GPU smoke 的冻结文件不再完全相同。**格式核对收据保存在独立整理工作区，真实模型证据仍归属于下述冻结恢复版。

恢复版已在固定 TRAIN 前 8 个来源完成真实 Qwen3.5-9B 基座推理：8 条 native 闭合，50 次模型 API 全部返回，6 次相同 grounded 请求在 API 前拦截。全部原生轨迹、token、EOS、源码及模型 SHA 的 CPU 审计通过，2892 个交付成员逐 SHA 校验通过。这是**已运行的隔离恢复版**的证据；本审阅副本新增了主干注释和测试组合，尚未单独再次运行 GPU，不应将两种文件身份混称。

固定 8 条中合法终止 0/8、预算耗尽 8/8。关键程序缺陷修复已获得原生验证，整体预算耗尽尚未解决，也没有由此得出类别准确率提升。旧 VAL422 扩大在发现重复请求缺陷后于 58/844 native 闭合、445 次 API 全部返回处主动停止，worker 退出 -15、driver 退出 1；其余 786 条未生成。这些负结果与原 422 分母原样保留，不当作完成的全量评估。

## 运行边界

不提高原预算、不使用 SFT adapter、不训练、不自动部署。输入事实只能从当前来源实际获取；模型解释和比较不升级为 raw fact。原引用、业务不足和终止验证继续有效。任何后续模型试验都应有新的冻结版本和独立收据；不为了已观察 VAL 分数调整 prompt、阈值或检查点。
