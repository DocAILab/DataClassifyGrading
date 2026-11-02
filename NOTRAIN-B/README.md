# NOTRAIN-B：VEH / FIN 验证集 CoreSkill

本实验对应项目文档 NONT-B。基座为 SiliconFlow 托管的 Qwen/Qwen3.5-9B，无 SFT、无 RL。固定两个模型型 Skill，不调用 Planner 或其他 Skill：

1. 根据 domain、field_name，从完整标准目录（类别路径）检索 5 个不同标准条目。
2. 读取上述 5 个条目的标准描述、参考标准与等级，选择 Top-1 并输出该条目对应的等级。

与汽车 NONT-D 对齐，初始输入仅有 domain、field_name；本流程不额外读取 field_description。候选按标准条目计数，同叶子不同等级可以有多个条目。模型输出必须是严格 JSON，不修复格式、不重试语义错误；仅网络错误、429 和 5xx 最多进行三次传输尝试。temperature=0，max_tokens=512，enable_thinking=false。托管权重版本和精度不可核实，零温度不保证跨运行逐字节一致。

数据固定为 DocAILab/DCG@84a9a462dcb828332f980ef121f40e1281ce1794 的 VEH Val397 和 FIN Val422。gold 只用于本地评估，绝不放入模型请求。失败、不合法输出均保留在完整验证集分母；宏指标按该验证集 GT 出现标签计算。分类采用完整非空类别路径。标准库提供等级映射，模型仍须显式输出 level；错级按错误计分。

`manifest.json` 记录算法、配置、源码及输入哈希；`VEH/`、`FIN/` 分别存储 inputs、registry、逐样本请求/响应/结果及 metrics.json；summary.json 为总结果。所有数据与结果仅供本地使用，不提交 Git。密钥仅通过 SILICONFLOW_KEY 传入子进程，不写入文件或日志。

在仓库根目录以 pwsh 运行：

```powershell
if (-not $env:SILICONFLOW_KEY) { $env:SILICONFLOW_KEY = [Environment]::GetEnvironmentVariable('SILICONFLOW_KEY', 'User') }
python NOTRAIN-B/run.py --workers 4
```

相同源码、配置和数据可恢复已完成样本；中断的样本不自动重新生成，以避免隐式择优重跑。运行进度见 progress.json 和终端日志。
