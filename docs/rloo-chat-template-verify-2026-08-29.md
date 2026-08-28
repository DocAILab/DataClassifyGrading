# Chat-Template Tool-Use 链路验证 — 2026-08-29

正常推理侧（transformers 直接推理，非 VeRL rollout）对**同一 agent-loop 链路**的
三模型对照验证。目的：把"模型行为"与"链路正确性"两个变量分离。

链路（与 verl 一致）：raw messages → `apply_chat_template(messages, tools=...)`
→ 模型生成 → `qwen3_coder` parser 解析 → 工具执行 → `role=tool` 回填 →
二次 `apply_chat_template` → 再生成。

---

## 1. 三组对照（同一样本，同一链路）

| 组 | 模型 | turn1（tokens/行为） | turn2（tokens/行为） |
|---|---|---|---|
| 基模 | 原始 Qwen3.5-9B（无 adapter） | 121：`<tool_call>search_categories` + **模板控制符泄漏**（`<|im_end|>`/`<|im_start|>user`/`<think>`）+ 第二个 tool_call | 52：**重复同一调用** + `<|endoftext|>`，**无终局 JSON** |
| w300 | 基模 + 300 步 direct SFT adapter | 121（**与基模逐字节相同**） | 52（相同，无终局） |
| step232 | 工具轨迹 SFT merged | 52（干净单 tool_call，无泄漏） | 115（继续 `get_category_examples`；正式 eval terminal_valid 217/219） |

- 三组 turn0 prompt 均为 **2881 tokens**（渲染一致：233 类 catalog + user 字段 + 空 think 块）；
- 三组工具调用均被 parser 正确解析并执行（链路三组全通）。

## 2. 结论

1. **链路正确**：`apply_chat_template`/tools 传入/parser/工具执行/回填/二次渲染
   在三组均正常工作——与 Qwen3.5 原生 XML 协议、verl `qwen3_coder` parser 三方一致；
2. **模板控制符泄漏是基模先验行为**，不是 w300 训练引入的异常；
3. **w300 ≈ 基模**：300 步 direct SFT（无工具轨迹）没有改变 agent-loop 行为
   （会输出工具调用，但工具结果后不给终局 JSON）；
4. **step232 是行为分水岭**：工具轨迹 SFT（四类确定性轨迹）教会
   "工具结果 → 终局 JSON"，正式 RLOO terminal_valid 99% 由此而来。

## 3. 因果链

```
基模：不会 agent-loop（模板泄漏 + 无终局）
  ↓ +300 步 direct SFT（无工具轨迹）
w300：行为 ≈ 基模
  ↓ +232 步工具轨迹 SFT
step232：学会 agent-loop（RLOO step300 eval 81.74%）
```

## 4. 证据文件（runtime-local，未入库）

| 文件 | 内容 |
|---|---|
| `tmp/chat-template-verify-base.json` | 基模组 A/B/C 完整证据（23KB） |
| `tmp/chat-template-verify-w300.json` | w300 组（23KB） |
| `tmp/chat-template-verify.json` | step232 组（23KB） |

每组含：A 输入 messages、B serialized prompt 全文、C 完整 trajectory
（assistant 生成文本、解析出的 tool calls、工具结果、二次生成文本）。
