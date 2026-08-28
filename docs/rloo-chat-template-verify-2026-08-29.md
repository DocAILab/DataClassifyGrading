# Chat-Template Tool-Use 链路验证 — 2026-08-29（修订版）

正常推理侧（transformers 直接推理，非 VeRL rollout）对**同一 agent-loop 链路**的
三模型对照验证。目的：把"模型行为"与"链路正确性"两个变量分离。

链路（与 verl 一致）：raw messages → `apply_chat_template(messages, tools=...)`
→ 模型生成 → `qwen3_coder` parser 解析 → 工具执行 → `role=tool` 回填 →
二次 `apply_chat_template` → 再生成。

> ⚠ 证据强度说明：三组对照目前各验证了**一条样本**（同一字段样本）。
> 单样本证据足以证明"链路在三组都能渲染/解析/执行工具"以及
> "step232 能正常完成轨迹"，但不能从单样本推出所有 rollout 的链路 100% 无问题。

---

## 1. 三组对照（同一样本，同一链路）

| 组 | 模型 | turn1（tokens/行为） | turn2（tokens/行为） |
|---|---|---|---|
| 基模 | 原始 Qwen3.5-9B（无 adapter） | 121：`<tool_call>search_categories` + 模板控制符泄漏（`<|im_end|>`/`<|im_start|>user`/`<think>`）+ 第二个 tool_call | 52：重复同一调用 + `<|endoftext|>`，无终局 JSON |
| w300 | 基模 + 300 步 direct SFT adapter | 121（与基模逐字节相同） | 52（相同，无终局） |
| step232 | 工具轨迹 SFT merged | 52（干净单 tool_call，无泄漏） | 115（继续 `get_category_examples`；正式 eval terminal_valid 217/219） |

- 三组 turn0 prompt 均为 2881 tokens（渲染一致：233 类 catalog + user 字段 + 空 think 块）；
- 三组工具调用均被 parser 正确解析并执行。

## 2. 结论（修订措辞）

1. **链路方面**：基模、w300、step232 已在同一 inference/agent-loop 链路下对照运行，
   且 step232 能正常完成完整轨迹——**目前没有证据表明 chat template、tool parsing 或
   tool execution 链路是主要根因，可以从主排查方向中降级**；
2. **模板标记泄漏方面**：在当前 chat template、prompt 和 decoding 设置下，原始
   Qwen3.5-9B 即可复现与 w300 相同的模板标记泄漏——**该现象不是 w300 SFT 引入的**
   （不上升到"Qwen3.5 固有行为"的普遍断言）；
3. **w300 ≈ 基模**：300 步 direct SFT（无工具轨迹）没有改变 agent-loop 行为；
4. **step232 是行为分水岭**：工具轨迹 SFT（四类确定性轨迹）教会了
   "observation → next action → terminal JSON" 的多轮状态机行为。

## 3. 核心因果判断

**工具调用能力 ≠ agent-loop 能力。**

```
基模 / w300：具备 tool-call syntax prior
  user → tool_call → tool result → 不知道下一步（继续调用 or 终局）
  ↓ 无工具轨迹 SFT
  RLOO rollout 大量 protocol failure → reward signal 极弱/接近 0
  （response_length=512 只是旁路的上限因素，不是根因）

step232：工具轨迹 SFT 教会完整状态机
  user → tool_call → tool result → 按轨迹继续决策 → terminal JSON
  terminal_valid ≈ 99%
  ↓ RLOO 在已掌握 agent protocol 的 policy 上优化分类策略
  step300 task accuracy = 81.74%
```

**response_length=512 重新定性**：从"根因"降级为"潜在性能限制"——它仍可能造成
长 reasoning / 第二次 tool call / terminal answer 被截断、复杂样本失败率升高，
但它解释不了"基模≈w300 而 step232 突然稳定"这一强干预结果。

## 4. 后续评估指标（不只看总 reward）

| 指标 | 含义 | 当前值（step300） |
|---|---|---|
| tool_call_valid_rate | 工具调用格式是否合法 | 438/438 = 100% |
| terminal_valid_rate | 能否走到合法终局 JSON | 219/219 = 100% |
| task_accuracy / exact_match | 最终分类分级是否正确 | 179/219 = 81.74% |

## 5. 证据文件（runtime-local，未入库）

| 文件 | 内容 |
|---|---|
| `tmp/chat-template-verify-base.json` | 基模组 A/B/C 完整证据（23KB） |
| `tmp/chat-template-verify-w300.json` | w300 组（23KB） |
| `tmp/chat-template-verify.json` | step232 组（23KB） |

每组含：A 输入 messages、B serialized prompt 全文、C 完整 trajectory。
