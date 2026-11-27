# 百炼金融教师轨迹

本次学生 SFT 使用的 [200 条教师轨迹、实际 Planner 请求和入选统计](../finance_m9/teacher200/teacher/README.md) 随训练主线发布。以下 25 条是此前的小规模回放样本。

这里保存金融分类分级项目的 25 条 TRAIN 字段轨迹，以及生成、回放和评分所用的教师代码。教师模型为 `qwen3.8-2.4t-a95b`。每个字段只保留一条最终轨迹。前 20 条沿用当时的采集结果，新增 5 条使用修复后的结果，各条使用的协议保存在自己的 `protocol.json` 中。

新采集的教师和 M9 学生共用[查表工具](runtime/collector/dsl_table_tools.py)。在训练元数据能唯一定位表时，`GET_TABLE_FIELDS` 只列出当前表其他字段的名称。查选中字段或查完整表上下文后，模型才会取得相应描述。规划器按需选择查证步骤。旧 25 条按各自的原协议回放，其中此前列作 SFT 候选的 13 条都引用过查字段名时提前出现的描述，需要在新协议下重采。

扩采中的 `selected_context_protocol` 版本把已选邻列的描述及证据编号列入分类请求，方便核对选中字段、业务主体和表记录的活动。教师和 M9 使用[同一证据整理函数](runtime/collector/selected_context_evidence.py)。每条旧轨迹继续按其采集时的协议回放。

## 最终分类

每个正常完成的字段都在 `decision.json` 的 `prediction` 中给出一个类别和等级。证据充分时采用分类器的判断。证据不足时直接采用首选类别，优先使用有效复核结果，其次使用检索第一名。没有有效排名时使用原有的标准目录排序。评分只比较这一个最终答案与参考标签。运行中断单独记为 `incomplete`。

| 字段范围 | 给出最终分类 | 类别与等级均正确 | 正确率 |
| --- | ---: | ---: | ---: |
| 新增 5 条，实验代码修复前 | 5/5 | 3/5 | 60% |
| 新增 5 条，实验代码修复后 | 5/5 | 5/5 | 100% |
| 累计 25 条 | 25/25 | 20/25 | 80% |

例如，091「TA投资人证件类型」的最终输出是「个人用户基本信息 / L3」，与参考一致。模型查完 42 个同表字段后仍无法完全排除机构投资人，判定文件用 `prediction_source: best_candidate` 记录这个答案的来源。[091 最终分类](trajectories/source-091/decision.json)

25 条中有 15 条由分类器依据证据作答，其中 13 条正确。其余 10 条采用首选类别，其中 7 条正确。SFT 人工复核沿用证据要求，共有 13 条候选。25 条共有 17 种完整工具顺序。[逐字段评分](scores.json) · [轨迹质量](quality.json) · [新增五条的前后对照](paired-scores.json)

## 先看一条轨迹

在 `scores.json` 的 `rows` 里按 `source_index` 或 `source_id` 找字段。每行的文件路径都相对于本目录。以 [255](trajectories/source-255) 为例。

| 文件 | 内容 |
| --- | --- |
| `trace.json.gz` | 原始原生轨迹 JSON 的完整字节，包括 Planner 尝试、被执行的工具、实际观测、证据引用、终止原因和最终状态。原始停止原因记录查证过程，供回放使用。最终对外结果见同目录的 `decision.json`。解压后的 SHA-256 在逐字段评分中。 |
| `calls.jsonl.gz` | 每行是一次主轨迹模型调用，含阶段、实际请求体及 messages、模型原文回复、输入输出摘要和生成用量。 |
| `decision.json` | 唯一的最终预测 `prediction`（类别、等级）。`prediction_source` 记录答案来自充分证据还是首选类别。 |
| `protocol.json` | 该条采集时使用的提示、证据和运行协议。 |
| `review.json` | 对于执行过独立复核的字段，保留复核请求体、模型回复和复核结果。 |

## 离线复核与重新采集

解压后进入有本 README 的目录。使用 Python 3.11 或更高版本，所需代码和样本已随包提供。下面的回放命令使用 Python 标准库，不需要 API key 或 GPU。

```shell
python -B replay.py
python -B replay.py 91 255
```

完整回放会重新执行原生控制器，并在每个实际模型调用位置对照当时的模型输入，再交回已记录的模型回复。25 条共对应 191 次主轨迹调用。另存的 10 次独立复核也会逐条校验。输出会核对终止状态、判定、评分和轨迹内容。

**重新采集**会向百炼发送新的计费请求。设置环境变量 `DASHSCOPE_API_KEY` 和 `DASHSCOPE_BASE_URL`。后者填写自己北京工作空间的 HTTPS 兼容接口根地址，结尾为 `/compatible-mode/v1`。选择本包中的字段编号，输出到一个全新的目录。新采集默认使用最终代码与 512 的思考预算，平台重新生成的内容可能与归档不同。

```shell
python -B runtime/collector/small_test.py runs/new-255 255
python -B runtime/collector/score_smoke.py runs/new-255 --references inputs/references.json
python -B runtime/collector/audit_smoke_paths.py runs/new-255/scores.json runs/new-255/quality.json
```

## 代码与数据入口

| 位置 | 用途 |
| --- | --- |
| [采集入口](runtime/collector/small_test.py) · [执行与回放协议](runtime/collector/smoke.py) | 串行采集、预算检查、规划与工具执行、合法终止检查。 |
| [证据查证](runtime/collector/evidence_followup.py) · [补查提示](runtime/collector/table_recovery.py) | 检索候选，提示尚未读到的同表列，取得新的可引用证据。 |
| [同表证据](runtime/collector/row_evidence.py) · [字段主体](runtime/collector/field_roles.py) | 去重呈现同表描述，比较相同属性前的不同限定语。 |
| [最终分类](runtime/collector/abstention.py) · [独立复核](runtime/collector/review_abstention.py) | 把证据充分的分类或证据不足时的首选落到同一个 `prediction` 字段。 |
| [回放入口](replay.py) · [逐条评分](runtime/collector/score_smoke.py) | 复现已记录的模型输入和原生轨迹，按唯一的最终类别和等级计算正确率。 |
| [本次输入](inputs) · [运行文件清单](runtime-sha256.json) | 输入包保存 25 个 TRAIN 起点和查询所需的元数据。另存的参考标签只用于离线评分。 |

`runtime/` 中的依赖提供当时实际使用的控制器、Skill 定义、证据校验器和百炼传输代码。发布版把原实验按训练边界筛选后的查询数据预先打包，省去对实验机历史目录的依赖。由此可以从一个干净目录重放所有 25 条，或用最终协议对这 25 个字段重新采集。
