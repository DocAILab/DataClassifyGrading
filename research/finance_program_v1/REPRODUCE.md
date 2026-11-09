# 金融主干运行与复现

先读 [实验协议](docs/EXPERIMENT_PROTOCOL.md)。在 `research/finance_program_v1` 工作；Python3.11+。原公开验证环境曾用Python3.13/Torch2.9 CPU。真实9B历史环境为Torch2.9、Transformers5.10.4、PEFT0.21、Accelerate1.15；这里只说明历史环境，不在import/test时安装或下载。

## 可直接运行：CPU测试

以下阶段使用合成/录制I/O，分别启动进程，避免同名 `gate`/`io_utils`/`program_loop` 混入另一个阶段：

```powershell
Set-Location teacher
python -B -m pytest tests -q -p no:cacheprovider --basetemp .test-tmp
Set-Location ../sft
python -B -m unittest discover -s tests -v
Set-Location ../evaluation
python -B -m unittest discover -s tests -v
Set-Location runtime
python -B -m pytest tests -q -p no:cacheprovider --basetemp .test-tmp
Set-Location ../..
python -B -m unittest discover -s publication_tests -v
```

每次pytest使用独占临时目录或新路径；Windows上系统旧pytest临时符号链接可能产生清理权限错误，不能因此改业务代码。CPU通过只证明契约/路由/回放/监督边界，不是9B语义准确率或GPU显存证据。

## 受控真实执行的顺序

1. 准备授权原始metadata、标准Registry、固定来源/分区/顺序、当前源码SHA、模型revision和逐文件身份；labels另存本地评估侧。公开 `study.json` 是合成fixture，不能替代这些资产。
2. Teacher采集使用 `teacher/collect.py`/`detached.py` 和版本化policy，密钥仅内存由调用方供给。占位config不是历史账本，不能直接复用付费额度。
3. 用 `sft/export_dataset.py`核200个来源与真实接受的Program；`token_audit.py --data ... --model ... --output ...`核无截断与响应+EOS监督。准备受控manifest、token审计和model identity后，才使用 `train_student.py --mode trial`，proof通过后fresh `--mode formal`。历史pipeline负责资源锁和实际退出；正式训练不会继承trial更新。
4. 评估使用 `evaluation/evaluate.py --cohort smoke`，然后按新的受控队列执行评估。原公开runner只提供smoke/val80接口；完整历史Val422补充协调器及私有配置未发布，不能把这个命令声称为一键422复现。两个arm均显式F66与相同study预算；只有sft的Planner开adapter。
5. 停写后运行 `evaluation/replay.py`、训练/路由身份gates，再本地评分、固定分母与独立复核。原run与manifest不可覆盖。

真实GPU/API命令需要资产和资源门全部满足；本次整理未执行这些步骤。各脚本的 `--help` 说明保留原参数。`pipeline.py`/`watch.py`是受控运行协作者，不能把缺失私有manifest当作公开CPU示例失败。

## 结果与可复现程度

`results/evidence-summary.json`是原422/844聚合；`results/model-roles.json`绑定teacher/第一版adapter和训练数据；`results/trajectory-export.json`记录1044条公开投影的大小/SHA。原结果文件和轨迹此次字节不变。`results/publication-verification.json`是原历史发布收据，不能当作本次新的测试结果；本次整理验证记录在 `docs/curation-verification.json`。

公开轨迹隐藏了私有ID/metadata、实际prompt、语义文本及推理，保留状态、Program投影、最终类别/等级和原trace摘要。它们不是模型训练数据，也不足以全轨迹逐字回放。完整数值复现需私有输入与标签、完整原调用及权重，现有公开代码不能独立恢复这些信息。
