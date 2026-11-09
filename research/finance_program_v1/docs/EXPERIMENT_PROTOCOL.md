# 金融 Program 主干实验协议

## 1. 对照问题与版本

主对照为原版 F66 `base` 与 `sft`。两臂用同一基座、输入来源、完整66个 Skill、标准库、预算与生成参数；只切换第一版 Planner LoRA。所有 Skill 模型调用均关闭 adapter，包括 SEARCH 和分类。`evaluation/backend.py` 以可信 Planner system 消息识别路由并保存层状态，生成失败后也恢复关闭。

代码整理起点为 `b6fde9b80802593ad8ab30bdfae66fc28483a849`。历史结果对应受控执行的冻结身份；整理后源码SHA会因注释排版改变，不能把新源码冒充旧运行的逐字源码。运行语义由整理前后 AST 比较和CPU测试核对。新真实运行必须建立自己的 manifest，不能覆写历史 manifest。

## 2. 数据、输入与目标

- Teacher 为百炼 `qwen3.8-2.4t-a95b`；发布的200条轨迹对应806次已核实 HTTP 请求。当前 `config.example.json` 是占位示例，其单次请求预算不代表原806次完整采集总预算。
- 学生使用固定200个训练来源，另14个 reserve 不参与训练。选择、实际原消息、接受的 Program 与来源覆盖由 `sft/export_dataset.py` 核对，不从公开脱敏轨迹反造训练数据。
- 学生监督仅为真实 Planner Program JSON + tokenizer EOS；prompt、Skill回应、元数据与评估标签不计算训练 loss。分词边界逐项验证，超长直接拒绝，不截断。
- 实际模型可见初始输入只有 domain/field_name；其他元数据由实际 Skill 取得。内部 source_id 是查找标识，不作为业务语义或答案。binder可为 `{}` 补齐已有 required输入，不能称模型独立生成了全部参数。
- `evaluation/data/study.json` 只有合成字段/类别，`protocol.json` 是公开 Planner system；它们不是完整Val422输入、参考标签或分区清单。

## 3. 第一版 Planner SFT

学生 `Qwen/Qwen3.5-9B`，revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`；接受的 adapter SHA `81b18e85c5ce9ceb9585cf069a87d389567a2be47198c35a0d79d73f850a2abc`。训练数据SHA见 `results/model-roles.json`。

`sft/train_student.py` 固定：1 epoch，200来源，microbatch=1，累积16，13次优化更新；AdamW，lr=1e-4、cosine、warmup3、weight_decay=.01、max_grad_norm=1。LoRA r8/alpha16/dropout.05，仅语言层40个实际目标模块（q_proj/v_proj/in_proj_qkv 后缀），base与vision冻结。BF16，三个精确身份GPU，27GiB/卡分片，梯度检查点；保留完整上下文，仅计算completion对应logits/loss，并将反向保存张量暂存CPU。不是全参数训练或Skill SFT。

先最长来源 trial，实际核有限 loss/gradient、LoRA更新、全部目标设备；formal必须重新加载新基座、核数据与试跑manifest，不能接着trial继续训练。正式期同时核200个来源各消费一次、13次有限梯度更新和只保存adapter。实际GPU身份/空闲证明属于受控manifest，不在占位示例中硬造。

## 4. F66 推理与预算

正式配置从受控study/config读取。公开合成study展示原实验预算：4 Planner尝试、3 Program、12 Skill步骤、单Program最多10步、相同调用限1。`ProgramConfig` 的通用构造默认值不同；主实验入口显式传study预算，不能用默认值替换并声称同协议。

评估seed在legacy中固定为20260928；SFT seed由受控manifest提供，不能仅凭评估seed补写未公开的训练配置。生成由 `evaluation/legacy/local_smoke.py` 与 `backend.py` 约束：greedy、max_new_tokens512、max_time90秒、上下文32768，超上下文拒绝且不截断。原模型EOS和监督tokenizer EOS取并集并核身份（248044/248046）。全模型和adapter FP16/cuda0冻结；正式训练资源与评估资源不同。

SEARCH由原版模型生成五个实际供给的短ID，再由执行器绑定标准；没有换成后续M9的三prefix概率排名。`evaluate.py` 每来源交替AB/BA，只有真实terminal guard通过才记合法终止。失败步骤的 `break` 结束当前Program，返回外层Planner；它不表示整个任务立刻成功或无限死循环。剩余预算为硬限。

## 5. 固定422分母、回放和评分

原正式结果固定422来源/臂、844轨迹。已核实阶段按原SHA复用，其余来源只补一次；smoke不进正式分母，未终止/错误不删。原阶段墙钟分别记录，平均来源耗时不是整轮墙钟。标签只在停写后本地评分，不能进入模型消息、候选选择或训练奖励。

合法类别/等级/联合、Top5首次/最后/任次、固定12类与实际8支持类Macro-F1、配对改善/退化及真实调用/token/时间分别报告。聚合见 `results/evidence-summary.json`；`evaluation/metrics.py` 是指标实现。所有公开结果/轨迹本次字节不改，不把一次整理包装成重跑。

完整复现需原授权study/registry、顺序/分区、真实完整调用/trace、model与adapter、source/config manifest及本地参考。原始控制器逐字回放和token/EOS/adapter路由证明保存在受控证据中。公开1044轨迹只保留脱敏投影，无法独立重建原始消息和语义事实。

## 6. 当前公开入口的能力边界

`evaluate.py --cohort` 保留历史 `smoke`/`val80` 队列接口；全Val422结果由历史受控补充分阶段执行完成。公开树没有私有补充协调器，不能把空smoke/val80合成fixture说成422复现。完整实验说明不等于仓库含有未公开资产。

Teacher当前policy源码有显式证据校准变体，主干保留原文件。重新采集必须核当前policy版本；历史Teacher200是否逐字等同某次新采集不能仅凭文件名推断，要核原消息/trace和训练数据SHA。

原版保留已知限制：通用feedback只保存最近一次Program observation；部分失败专用账本单独保留。Planner仍可能在缺输入、no_match或坏输出后重复规划；静态计划最大长度与实际剩余Skill额度并非完全同一检查。原公开base231/422未合法终止，不能把原版称为已修预算bug。本次没有合入隔离recovery补丁、重训或改prompt追分。

Val422是已观察开发集，239 original/183 synthesized不等于专家Gold/全错，多个字段可能共享上下文；source级结果不是422个独立业务族的证据。标准映射出的等级不等于独立模型分级推理，合法引文也不自动证明语义充分。
