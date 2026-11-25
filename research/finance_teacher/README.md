# 百炼金融教师轨迹

这里保存金融分类分级项目的 25 条 TRAIN 字段轨迹，以及生成、回放和评分所用的教师代码。教师模型为 `qwen3.8-2.4t-a95b`。每个字段只保留一条最终轨迹。前 20 条沿用当时的采集结果，新增 5 条使用修复后的结果，各条使用的协议保存在自己的 `protocol.json` 中。

## 结果

| 字段范围 | 正式给出分类 | 类别与等级均正确 | 弃权首选命中 | 含建议的总命中 |
| --- | ---: | ---: | ---: | ---: |
| 新增 5 条，修复前 | 2/5 | 1/5 | 2/3 | 3/5 |
| 新增 5 条，修复后 | 2/5 | 2/5 | 3/3 | 5/5 |
| 累计 25 条 | 15/25 | 13/25 | 7/10 | 20/25 |

25 条共有 17 种完整工具顺序。12 条与至少另一条走了相同顺序。13 条正式答对的轨迹可进入 SFT 人工复核，其中 10 条的完整工具顺序有重复。[逐字段评分](scores.json) · [路径、引用和相似度](quality.json) · [新增五条的前后对照](paired-scores.json)

- **255 待办表序列号**从弃权变为正式答对。把可引用的同表字段去重后，模型结合「是否启用」和「待办页面跳转链接」识别出信息系统配置数据。固定原始输入后，单独重测分类环节也得到相同结论。[固定输入结果](fixed-255.json)
- **091 TA投资人证件类型**原先被判成人力资源数据。新版把「TA投资人」和「系统内职工」两个限定语列出来。查完 42 个同表字段后仍难以区分个人与机构，最终弃权，首选个人用户基本信息 / L3。[实际判定](trajectories/source-091/decision.json)
- **249 货币业务类别**的复核回复缺少必需字段，程序采用当次检索的第一名债券数据 / L1。该建议命中参考标签，保留弃权状态。[实际判定](trajectories/source-249/decision.json)

## 先看一条轨迹

在 `scores.json` 的 `rows` 里按 `source_index` 或 `source_id` 找字段。每行的文件路径都相对于本目录。以 [255](trajectories/source-255) 为例。

| 文件 | 内容 |
| --- | --- |
| `trace.json.gz` | 原始原生轨迹 JSON 的完整字节，包括 Planner 尝试、被执行的工具、实际观测、证据引用、终止原因和最终状态。解压后的 SHA-256 在逐字段评分中。 |
| `calls.jsonl.gz` | 每行是一次主轨迹模型调用，含阶段、实际请求体及 messages、模型原文回复、输入输出摘要和生成用量。 |
| `decision.json` | 正式预测或弃权，以及弃权时首选建议的来源。 |
| `protocol.json` | 该条采集时使用的提示、证据和运行协议。 |
| `review.json` | 对于执行过独立复核的字段，保留复核请求体、模型回复和复核结果。 |

## 离线复核与重新采集

解压后进入有本 README 的目录。使用 Python 3.11 或更高版本，所需代码和样本已随包提供。下面的回放命令使用 Python 标准库，不需要 API key 或 GPU。

```shell
python -B replay.py
python -B replay.py 91 255
```

完整回放会重新执行原生控制器，并在每个实际模型调用位置对照当时的模型输入，再交回已记录的模型回复。25 条共对应 191 次主轨迹调用。弃权后的独立复核记录另存。输出会核对终止状态、判定、评分和轨迹内容。

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
| [弃权判定](runtime/collector/abstention.py) · [独立复核](runtime/collector/review_abstention.py) | 在保留正式弃权状态的同时输出首选建议。 |
| [回放入口](replay.py) · [逐条评分](runtime/collector/score_smoke.py) | 复现已记录的模型输入和原生轨迹，分开统计正式命中和建议命中。 |
| [本次输入](inputs) · [运行文件清单](runtime-sha256.json) | 输入包保存 25 个 TRAIN 起点和查询所需的元数据。另存的参考标签只用于离线评分。 |

`runtime/` 中的依赖提供当时实际使用的控制器、Skill 定义、证据校验器和百炼传输代码。发布版把原实验按训练边界筛选后的查询数据预先打包，省去对实验机历史目录的依赖。由此可以从一个干净目录重放所有 25 条，或用最终协议对这 25 个字段重新采集。
