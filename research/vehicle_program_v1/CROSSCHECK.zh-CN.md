# 与汽车 NONT-A/B/C 合表前的核验清单

状态：待双方核对，不能据本清单宣称已经完成公平对照。

| 项目 | 本次NONT-D | 需A/B/C确认 |
|---|---|---|
| 数据 | DocAILab/DCG@84a9a462，Val397 | 相同revision、ID集合和分母；不混Test395 |
| 输入权限 | 初始domain/field_name，按需field_description | 模型不得看到分类、分级或样本顶层标准匹配信息 |
| 标准 | 59条目/44叶子，category内部为条目ID | Top-5按条目还是去重类别；同叶子多等级如何处理 |
| 模型 | SiliconFlow Qwen3.5-9B，revision/精度未核实 | 是否同服务和模型；本地FP16不能默认等价 |
| 生成 | 32K上下文，512输出，temperature=0，thinking=false | 相同或明确披露差异 |
| C/D控制 | 同F66；D最多3Program/4规划/12Skill，每Program10步 | 共同Skill后端、总执行预算、工具版本和反馈规则 |
| 主评分 | 路径分类、分级、联合；失败全部入397 | 独立重算，合法终局率不冒充类别取值合法率 |
| F1 | 支持14叶子及固定44叶子分别报告 | 标签全集、无效预测处理；不照搬固定56联合标签 |
| 来源历史 | 9先导＋388后续，显式基础设施恢复 | 先导是否纳入、失败是否重跑、费用/耗时口径 |

A=直接回答；B=固定Top-5→Top-1两个核心Skill；C=逐步选Skill；D=有序Program及预算内重规划。四组均未训练。C/D是编排方式对照，D/后续TRAIN-A是Planner SFT对照。

项目指标表中Weighted-F1、分级/联合Macro-F1、Macro-Precision/Recall及answer/level解析率等，需按双方统一评测定义补齐；本次不将Program validated_terminal强行改名为另一种输出schema的解析率，也不复制金融固定标签数。
