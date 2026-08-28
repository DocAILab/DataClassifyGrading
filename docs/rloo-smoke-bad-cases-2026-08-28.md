# RLOO Smoke Bad Cases — 2026-08-28

来源：8-27 RLOO smoke（早期 SFT adapter 初始化、3 步训练、64 样本、agent-loop rollout 打印）。

---

## 1. 统计

| 项 | 值 |
|---|---|
| 总 rollout 记录 | 50 |
| **bad（valid=False）** | **45（90%）** |
| valid（严格终局 JSON） | 5（10%） |
| 样本 ground truth | **单一 gt**（smoke 只评估了单一样本的重复采样） |

> ⚠ 注意：50 条 rollout 全部来自同一 ground truth——bad case 集合只能说明"该样本下模型行为"，不能推广到全分布。

## 2. 失败形态分类（45 条）

| failure_type | 数量 | 说明 |
|---|---|---|
| `tool_call_closed` | **31** | 输出完整闭合 `<tool_call>...</tool_call>`（工具调用可被解析执行），但轨迹结束在工具调用，**无终局 JSON** |
| `free_text_reasoning` | **8** | 输出自然语言推理文本，无 tool_call、无终局 JSON |
| `tool_call_partial_or_malformed` | **6** | 工具调用截断/畸形（缺起始符、未闭合、参数截断），解析失败 |

## 3. 机制解读

- verl `qwen3_coder` parser 用正则识别**文本** `<tool_call>`（Qwen3.5 原生 XML 协议，非 special token）——`tool_call_closed` 类会被正确识别并执行工具；
- 失败根因不是"框架不识别"，而是 **smoke 起点 checkpoint（早期 SFT、数据无工具轨迹）在工具循环后不输出终局 JSON**：
  - `free_text_reasoning`：无 tool_call → 框架判定终止（无终局）；
  - `tool_call_closed`：工具执行后模型继续输出工具调用，直到轮次/预算耗尽；
  - `tool_call_partial_or_malformed`：畸形调用 → 解析失败 → 提前终止。
- 对比：正式 RLOO（工具轨迹 SFT 起点）step90 的 terminal_valid = 217/219（99%），格式问题已解决；bad cases 转为"预测错"为主（48/49）。

## 4. 与正式 RLOO bad case 的关系

- smoke bad：**格式/行为失败为主**（45/45 无终局 JSON）；
- 正式 RLOO bad：**预测错为主**（48/49 格式对、答案错）；
- 两个阶段的瓶颈完全不同：smoke 是"不会用工具循环"，正式 RLOO 是"分类准确率"。
