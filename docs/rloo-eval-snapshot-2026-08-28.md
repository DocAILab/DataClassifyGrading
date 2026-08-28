# RLOO Eval Snapshot — 2026-08-28

Qwen3.5-9B RLOO（shougang 域）阶段性评估快照。

---

## 1. 配置（协议层）

| 项 | n=4 段 | n=2 段 |
|---|---|---|
| rollout.n | 4 | 2 |
| train_batch_size | 8 | 8 |
| prompt/response 上限 | 6144 / 512 | 6144 / 512 |
| max_model_len | 8192 | 8192 |
| attention | SDPA | SDPA |
| enable_thinking | False（⚠ 见 §5） | False（⚠ 见 §5） |
| data.shuffle | False | False |
| lr / LoRA | 1e-5 / r8 α16 | 1e-5 / r8 α16 |
| 起点 | merged SFT step232 | n4 global_step_90 |

## 2. Eval 结果（219 条 held-out；协议一致：seed 42 / max_tokens 512 / max_seq_len 8192 / eager / FlashInfer off）

| 指标 | SFT step232 | RLOO step90 (n=4) | RLOO step150 (n=2) |
|---|---|---|---|
| exact reward | 21/219 = **9.59%** | **173/219 = 78.99%** | **170/219 = 77.63%** |
| terminal valid | 35/219 = 16.0% | 217/219 = 99.1% | 218/219 = 99.5% |
| tool calls（成功/失败） | 240 (240/0) | 438 (435/3) | 439 (436/3) |
| zero reward | 198 | 46 | 49 |

**78.99% vs 77.63%：-3 条（-1.36pp），在 219 条样本的噪声范围（±~5.5% CI）内，不能判定 n=2 退化。**

## 3. Bad case 分类（step150）

| 类别 | 数量 |
|---|---|
| 总 bad（reward=0） | 49 |
| 格式错（terminal_answer_valid=false） | 1 |
| 预测错（格式对、答案错） | 48 |

- 唯一格式错样本特征：response_tokens 触顶（512 预算耗尽）+ 1 次工具失败 → 无终局 JSON（`response_length=512` 截断风险实例）；
- 两轮（step90/150）bad 高度重合（46/49 相同）——稳定 hard cases；
- **瓶颈是分类准确率**（48/49 格式对但答案错），不是 agent-loop 行为。

## 4. 暴露的问题清单（方法论层）

### P0 配置一致性
1. **thinking 分叉**：SFT 数据带 reasoning_content（think 内容，数据线决策），但 RLOO 训练/评估 `enable_thinking=False`（从早期 smoke 继承，未按"方向 2"更新）。当前成绩是关 think 下取得的，与 SFT 数据不一致。
2. **n 混杂**：step90=n4、step150+=n2，两轮 eval 混两个变量，无法干净归因。

### P1 复现性
3. 正式 RLOO 启动器（含 --resume/--n-rollout 扩展）不在 git，属 runtime-local；git 内 launcher 与服务器运行版存在 thinking 决策分叉。
4. verl site-packages patches 与 git patch 文件的一致性未逐字节比对。
5. 训练/评估所需 registry/corpus/manifest 为 runtime-local 资产（不入库）。

### P2 训练信号
6. 训练 reward 饱和（pg_loss≈0、梯度由 KL 驱动）、entropy 低——训练 reward 不可作泛化证据；训练/泛化差距大（train ~1.0 vs held-out 77–82%）。
7. `response_length=512` 定性为**潜在性能限制**（非根因）：可能截断长 reasoning / 第二次 tool call / terminal answer（§3 唯一格式错样本为实证），但不解释"基模≈w300、step232 突然稳定"的强干预结果。

### P3 数据/评估
8. 219 条测试集偏小（CI ±~5.5%），77.63 vs 78.99 不显著；未做 F1/更大集对比。
9. eval 未持久化 raw trajectory——bad case 只有统计字段，原始生成文本需重跑才能获得。

## 5. 与 RLOO smoke bad cases 的关系

- smoke（早期 checkpoint）bad：**格式/行为失败为主**（无终局 JSON：工具调用收尾/推理文本泄漏/截断畸形）；
- 正式 RLOO bad：**预测错为主**（48/49 格式对、答案错）；
- 两个阶段瓶颈完全不同：smoke 是"不会用工具循环"（工具调用能力 ≠ agent-loop 能力，基模先验仅有 tool-call syntax），正式 RLOO 是"分类准确率"；
- 链路（chat template / tool parsing / tool execution）经三模型对照（基模/w300/step232，见 `docs/rloo-chat-template-verify-2026-08-29.md`）——目前无证据表明是主要根因，已从主排查方向降级。
- 详见 `docs/rloo-smoke-bad-cases-2026-08-28.md`。
