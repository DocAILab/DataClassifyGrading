# RLOO Smoke Bad Cases — 2026-08-28

来源：`/root/autodl-tmp/rloo-smoke/rloo-smoke.log`（8-27 RLOO smoke：adapter-w300 初始化、3 步训练、64 样本、AgentLoopWorkerTQ `[SMOKE]` 打印，行 1191–1231）。

完整数据：`tmp/smoke-bad-cases-2026-08-28.json`（50 条，含 `gt` / `valid` / `text_tail` 原文 / `failure_type`）。

---

## 1. 统计

| 项 | 值 |
|---|---|
| 总 rollout 记录 | 50 |
| **bad（valid=False）** | **45（90%）** |
| valid（严格终局 JSON） | 5（10%） |
| 样本 ground truth | **全部为 `B5-2-2/L2`（单一 gt）** |
| 对应 train.parquet source | gt 多义（437 条同 gt），无法唯一关联 source_id |

> ⚠ 注意：smoke 的 50 条 rollout 全部来自同一 ground truth——bad case 集合只能说明"该样本下模型行为"，不能推广到全分布。

## 2. 失败形态分类（45 条）

| failure_type | 数量 | 说明 |
|---|---|---|
| `tool_call_closed` | **31** | 输出完整闭合 `<tool_call>...</tool_call>`（工具调用被正确解析执行），但轨迹结束在工具调用，**无终局 JSON** |
| `free_text_reasoning` | **8** | 输出自然语言推理文本（"Let's check...", "The category X matches..." 等），无 tool_call、无终局 JSON |
| `tool_call_partial_or_malformed` | **6** | 工具调用截断/畸形（`<` 缺失、`</tool_call>` 未闭合、参数截断），解析失败 |

## 3. 样例（text_tail 尾部，已截断展示）

**① tool_call_closed（31 条，典型）**：
```
<tool_call>
<function=browse_categories>
<parameter=prefix>
B
</parameter>
</function>
</tool_call>
```
→ 工具调用格式完整、会被 parser 识别并执行；但模型在预算/轮次内**没有继续输出终局 JSON**。

**② free_text_reasoning（8 条，典型）**：
```
...Let's check a few other related categories manually, such as:

- **93|表面缺陷分析** (Surface defect analysis)
- **94|过程质量分析** (Process quality analysis)
- **83|表面质量判定处置...
```
→ 思考文本直接泄漏到输出（基础模型先验行为），无 tool_call → 直接 TERMINATED → reward 0。

**③ tool_call_partial_or_malformed（6 条，典型）**：
```
tool_call>
<function=search_categories>
<parameter=field_name>
ABNR_CODE_1
...
```
→ 开头缺 `<`（截断）或 `</tool_call>` 未闭合 → parser 解析失败 → TERMINATED。

**④ valid（5 条）**：
```
{"answer": "83", "level": "L3"}
```
→ 严格终局 JSON（parse 通过、reward=1）。

## 4. 机制解读（与协议验证一致）

- verl `qwen3_coder` parser 用正则识别**文本** `<tool_call>`（Qwen3.5 原生 XML 协议，非 special token）——① 类会被正确识别并执行工具；
- 失败根因不是"框架不识别"，而是 **adapter-w300（300 步 SFT、数据无工具轨迹）在工具循环后不输出终局 JSON**：
  - ② 类：无 tool_call → 框架 TERMINATED（无终局）；
  - ① 类：工具执行后模型继续输出工具调用，直到 `max_assistant_turns=4` 或 512 token 预算耗尽；
  - ③ 类：畸形调用 → 解析失败 → 提前 TERMINATED。
- 对比：正式 RLOO（step232 起点，工具轨迹 SFT）step90 的 terminal_valid = 217/219（99%），格式问题已解决；bad cases 转为"预测错"为主（48/49）。

## 5. 与正式 RLOO bad case 的关系

- smoke bad：**格式/行为失败为主**（45/45 无终局 JSON）；
- 正式 RLOO bad：**预测错为主**（48/49 格式对、答案错）——两个阶段的瓶颈完全不同：smoke 是"不会用工具循环"，正式 RLOO 是"分类准确率"。
