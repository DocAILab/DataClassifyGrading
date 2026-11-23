# 金融 Val422 复现数据

本目录收录同一组 422 个金融字段在两次实验里的输入和结果。可按 `source_id` 对齐。M9 使用 fresh162 规划器和 Scale1169 分类器，得到 **185/422** 个类别匹配。未微调的 Qwen3.5-9B 使用原版 F66 流程，得到 **79/422** 个类别匹配。两次实验的执行协议不同，应各自复核。

| 要找的材料 | 文件与用途 |
| --- | --- |
| Val422 参考标签 | [val422-references.json](val422-references.json) 以 `source_id` 为键，含完整类别路径 `category` 和敏感等级 `grading_level`。直接传给 M9 的 `--references`。 |
| M9 实际输入 | [val422-inputs.json](val422-inputs.json) 保留归档顺序，只含来源 ID 和可观察的元数据。直接传给 `--data`。 |
| M9 逐字段评分 | [m9-185-per-field.csv](m9-185-per-field.csv) 每行对应一个字段，含参考与预测的类别、等级、终止状态、匹配标记和停止原因。 |
| 固定流程的依据 | [m9-first-plans.json](m9-first-plans.json) 按首次实际输出的完整 Program 分组，含工具顺序、参数及来源 ID。四种计划分别覆盖 401、14、5、2 个字段。401 个字段都选择了文件中的第一种八步计划。固定执行这套计划的独立对照仍待实现和评估；185/422 来自自主规划。 |
| 79/422 使用的输入与配置 | [原始 study.json](nontraining-79/study.json) 含元数据、分类标准、执行预算和分组顺序。[config.json](nontraining-79/config.json) 摘录实际基座版本、精度、生成参数和适配器状态；[原版提示协议](../../finance_program_v1/evaluation/data/protocol.json) 与实验归档内容一致。 |
| 79/422 逐字段评分 | [nontraining-79/per-field.csv](nontraining-79/per-field.csv) 用原始轨迹哈希将已发布的 [nontraining 轨迹](../../finance_program_v1/trajectories/student_val422_base.redacted.jsonl) 对回来源 ID，并逐项核对历史评分。正式分母为原有的 80 个字段加补评的 342 个；工程冒烟运行不另计。 |

## 复算

在仓库根目录执行以下命令，只需 Python 标准库。脚本读取两份预测和独立标签，逐行重新判断类别、等级、联合匹配，并核对来源、评分标记和汇总。

```bash
python -B research/finance_m9/val422/verify.py
```

| 实验 | 字段数 | 合法终止 | 类别匹配 | 等级匹配 | 类别与等级均匹配 |
| --- | ---: | ---: | ---: | ---: | ---: |
| M9 | 422 | 410 | 185 | 230 | 185 |
| Nontraining | 422 | 191 | 79 | 111 | 79 |

分母保留全部 422 个字段。未合法终止的记录按未匹配计。重新生成 M9 预测需要基座和两套 SFT 权重，运行命令见 [M9 README](../README.md)。

## 数据版本

本次使用金融验证集的实验归档版本，原始 `DCG_FIN_VAL.json` 的 SHA-256 为 `1f06268f8c497fd67adaf0e4529a14092f1595df6531b9366bc03d160e18245c`。敏感级将“常规一般数据”“敏感一般数据”“重要数据”分别映射为 L1、L2、L3。

与项目在 Hugging Face 发布的指定版本核对后，422 个来源 ID、完整类别与等级全部相同。一个字段的名称在归档中为 `ORIGINAL`，在公开版本中为 `synthesized`。归档中 239 条的标签来源标为 original、183 条标为 synthesized；该公开版本将 422 条均标为 synthesized。本目录保留模型当时实际看到的输入。评分衡量与数据集参考标签的一致程度。[公开金融验证集（固定版本）](https://huggingface.co/datasets/DocAILab/DCG/blob/024900be35a3ff06ff149615e94fa43614f2656c/DCG_FIN/DCG_FIN_VAL.json)
