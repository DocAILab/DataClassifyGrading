# 金融分类分级训练主线（M9）

一个字段进来时，规划器先看到字段名，再决定要查哪些元数据。检索模块从 12 个金融类别中选出 5 个候选，分类器根据已经查到的证据选择类别并给出引用。控制器检查引用和终止条件，按选中标准条目给出等级。证据不足时，流程可以弃权。

同一个 Qwen3.5-9B 基座挂两套 LoRA 权重。**fresh162 负责规划，Scale1169 负责分类。GRPO 只更新规划权重。** 已有两套 SFT 权重时，可以直接评估或开始 RL。

复核实验可直接使用仓库内的 [Val422 输入、参考标签、逐字段评分和 nontraining 运行配置](val422/README.md)。其中也提供了 422 个字段的实际首计划，方便核对固定流程对照的设计。

## 代码架构

| 环节 | 输入、职责与输出 | 实现 |
| --- | --- | --- |
| Planner SFT | 162 条按原顺序排列的示范。监督完整 Program JSON 和 EOS，Program 可含 1–16 步。一轮训练共 11 次更新 | [train_sft.py](train_sft.py)、[示范校验和分词](../finance_rl_v1/experiments/dependencies/finance_program_student_fresh162_20261022_r5/dataset_contract.py) |
| 分类 Skill SFT | 1,169 个训练来源，每条展示全部 12 个选项。仅类别 ID 的 token 参与损失，一轮训练共 147 次更新 | [train_sft.py](train_sft.py)、[类别 ID 的监督区间](partial_contract.py) |
| 元数据与执行器 | 从当前数据集的字段名构造初始状态。规划器按需取数，在相同数据库、相同表的范围内使用上下文。每条记录最多有 20 次规划尝试、20 个 Program、32 次 Skill 执行，每个 Program 最多 16 步 | [episode.py](episode.py)、[数据投影](native_projection.py)、[原生控制器](full15/native/native_env.py) |
| 模型调用 | Planner 选 fresh162。所有 `SEARCH_STANDARD_*` 用冻结基座给完整类别 ID 算概率，用 3 次前向计算排出 Top 5。`INFER_CATEGORY` 和 `INFER_CATEGORY_FROM_TOPK` 选 Scale1169，其余模型型 Skill 走冻结基座。Scale1169 的训练任务是 `INFER_CATEGORY`，两个分类接口在推理时共用它 | [路由](routing.py)、[模型调用](model.py)、[检索输入](native_rank_input.py)、[候选排序](rank_contract.py) |
| 评估 | 使用贪心生成。逐条记录是否通过原生终止校验、最终类别与等级、停止原因和调用数 | [run.py](run.py) 的 `evaluate` |
| Planner RL | 同一来源采样多条计划，分类与检索沿用上面的实现。合法终止且类别、等级全部匹配得 1 分，其余业务结果得 0 分。对组内回报不同的轨迹计算优势，只在 Planner 实际生成的 token 上更新，另加载一份冻结的初始 Planner 作为 KL 参考 | [run.py](run.py) 的 `train-rl`、[grpo.py](grpo.py) |

`native_assets.py` 把仓库内已有的执行器和 Skill 定义组装到本次运行的临时目录，逐一核对原生执行器清单中的 121 个文件，再写入当前数据集的同表范围。独立的 [nontraining 入口](../finance_program_v1/nontraining/README.md) 继续使用它自己的执行路径。

## 环境与输入

在 `research/finance_m9` 下执行以下命令。CPU 检查需要 Python 3.10+ 和 PyTorch。实际模型运行还需支持 BF16 的 CUDA GPU、支持 Qwen3.5 的 Transformers、PEFT、Accelerate 和 safetensors。默认把模型放在第一块可见 GPU。基座、适配器和数据均从本地路径读取。

基座使用 `Qwen/Qwen3.5-9B`，revision 为 `c202236235762e1c871ad0ccb60c8ee5ba337b9a`。[文件校验值](assets/model-hashes.json) 随代码提供。两套实验权重的 `adapter_model.safetensors` SHA-256 分别为：

| 权重 | SHA-256 |
| --- | --- |
| fresh162 Planner | `6d78e440ab0c9749427e002d54fd94cca06dea77c31b63ae3628b48fdbec4e64` |
| Scale1169 分类 Skill | `0e52b9439b0016b0cad2903392030fca4b501019c7378a3a594b0fb6d1d4faf7` |

评估和 RL 各读一份元数据 JSON。下例是一个合成的训练输入。验证集使用 `"split": "VAL"`。支持的元数据键见 [输入白名单](resident_input_contract.py)，可省略未知字段。训练集和验证集应预先按来源划分。

```json
{"split":"TRAIN","rows":[{"source_id":"demo-1","metadata":{"field_name":"coupon_rate","field_description":"合成样例的债券票面利率"}}]}
```

参考答案放在单独的 JSON，以相同 `source_id` 对齐。类别使用 [金融标准](assets/study.json) 中的完整键。RL 在轨迹结束后读对应答案打分。评估时提供 `--references` 可计算类别与等级联合匹配数。

```json
{"split":"TRAIN","references":{"demo-1":{"category":"finance:业务数据/金融市场数据/债券数据","grading_level":"L1"}}}
```

## 运行

**1. 检查执行器与参数更新。** CPU 测试使用合成字段和微型张量，覆盖一份七步计划、证据不足弃权、输入隔离，以及一次真实优化器更新的参数范围。

```bash
python -B -m unittest -v test_mainline
```

**2. 准备两套 SFT 权重。** 已持有上述两套适配器时，直接进入第 3 步。需要重训时，两个命令各自从同一个基座开始。以下示例从仓库根目录运行 `cd research/finance_m9` 后执行；大写变量代表本地路径。

```bash
python train_sft.py planner --model "$MODEL" --data "$TRAIN162_JSONL" --output "$PLANNER"
python train_sft.py skill   --model "$MODEL" --data "$SCALE1169_JSON"  --output "$SKILL"
```

Planner 输入为原始的 `train162.jsonl`，每行有 `id`、`messages` 和 `completion`。分类 SFT 输入为 `{"split":"TRAIN","rows":[...]}`，每行有 `source_id`、`metadata` 和 `target_option_id`（如 `C001`）。从实验归档 `cohort-inputs.json` 的 `train` 数组按原顺序取这三项即可。代码校验示范和分类训练集的身份。重训产物带有记录实际权重摘要的 `sft.json`，可直接传给后续命令。

**3. 评估 SFT 主线。**

```bash
python run.py evaluate --model "$MODEL" --planner "$PLANNER" --skill "$SKILL" --data val422/val422-inputs.json --references val422/val422-references.json --output runs/sft-val
```

**4. 训练并评估 RL 规划器。** 默认每个来源采样 4 条轨迹，每 8 个来源形成一个更新批次。每个批次使用当时的规划器采样，只更新一次；全部优势为零时跳过更新。模型调用错误会终止运行，防止半条轨迹进入训练。

```bash
python run.py train-rl --model "$MODEL" --planner "$PLANNER" --skill "$SKILL" --data "$TRAIN_JSON" --references "$TRAIN_REFS" --output runs/rl
python run.py evaluate --model "$MODEL" --planner "$PLANNER" --skill "$SKILL" --candidate runs/rl/step-001/planner --data "$VAL_JSON" --references "$VAL_REFS" --output runs/rl-val
```

每次运行使用新输出目录。评估产出 `predictions.jsonl` 和 `completion.json`。RL 产出 `episodes.jsonl`，每次有效更新保存 `step-NNN/planner/` 和 `checkpoint.json`。评估时将示例中的 `step-001` 换成实际生成的步骤。检查点会校验起始 Planner 和冻结分类器的权重身份。

本次交付包含训练、评估代码和 CPU 验收。9B 基座、两套适配器及业务训练数据需另行准备。M9 接 GRPO 后的完整训练和 Val422 评分仍待运行。
