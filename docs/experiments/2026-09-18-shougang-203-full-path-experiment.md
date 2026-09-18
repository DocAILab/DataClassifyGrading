# Shougang-5415 203 类完整路径对照实验

## 目的

在保持官方 `shougang-5415` train/val/test 划分不变的前提下，将类别身份从已发布的 193 个 `level_4` 文本改为完整的
`level_1 > level_2 > level_3 > level_4` 路径，用于判断“——”折叠是否简化了任务。

## 实现

`script/canonical/import_dcg_shougang.py` 新增：

```text
--identity-mode level4     原有 193 类口径（默认）
--identity-mode full-path  完整四级路径口径（预期 203 类）
```

`prepare_shougang_5415(..., identity_fields=PATH_FIELDS)` 会：

- 保留原始 `category_path` 作为标签源信息；
- 使用四个分类字段生成唯一 `category_id`；
- 严格校验官方 split 为 4,332/541/542；
- 严格校验完整路径类别数为 203；
- 拒绝验证集或测试集中出现训练集未覆盖的路径。

## 本地数据准备结果

数据源仍是 [DocAILab/DCG](https://huggingface.co/datasets/DocAILab/DCG/tree/main) 的
`shougang/shougang-5415`，不是 `xianyu9n/sg_sft`。本地已生成：

- runtime：`results/dcg-shougang-5415-full-path-runtime-20260918/`
- SFT parquet：`results/dcg-shougang-5415-full-path-sft-20260918/`

实际结果：

| 项目 | 结果 |
|---|---:|
| 总记录 | 5,415 |
| train / val / test | 4,332 / 541 / 542 |
| 完整路径类别 | 203 |
| train 未覆盖的 val/test 路径 | 0 |
| SFT train parquet 行数 | 8,664 |
| SFT val parquet 行数 | 1,082 |
| SFT test parquet 行数 | 1,084 |
| SFT exporter 验证 | passed |

SFT exporter 仍采用与 193 类 baseline 相同的四类元数据：
`field_name`、`table_name`、`field_description` 和 `table_description`，因此后续可以将差异归因于类别身份定义，而不是输入元数据改变。

## 后续 GPU 实验

当前已完成 CPU 数据准备，还没有把旧的 193 类 checkpoint 当作 203 类模型评测。两者的 `category_id` 和候选提示不同，必须在这份 203 类 SFT parquet 上重新训练。

正式对比应保持：

- 相同 Qwen3.5-9B 基座模型、LoRA 超参数和一个 epoch；
- 相同 542 条测试样本、解码约束和贪心推理设置；
- 同时报告 193 类和 203 类的 Stage-1 Recall@5、类别准确率、等级准确率和严格联合准确率。

