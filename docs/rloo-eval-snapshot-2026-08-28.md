# RLOO Eval Snapshot — 2026-08-28

Qwen3.5-9B RLOO（shougang 域）阶段性评估快照。
所有数值均来自服务器持久化产物（见文末 provenance）。

---

## 1. 配置与运行状态

### RLOO 训练
| 项 | n=4 段 | n=2 段（当前） |
|---|---|---|
| rollout.n | 4 | 2 |
| train_batch_size | 8 | 8 |
| prompt/response 上限 | 6144 / 512 | 6144 / 512 |
| max_model_len | 8192 | 8192 |
| attention | SDPA | SDPA |
| enable_thinking | **False**（⚠ 见 §5） | **False**（⚠ 见 §5） |
| data.shuffle | False | False |
| lr / LoRA | 1e-5 / r8 α16 | 1e-5 / r8 α16 |
| 起点 | merged SFT step232 | n4 global_step_90 |

---

## 2. Eval 结果（219 条 held-out，协议一致：seed 42 / max_tokens 512 / max_seq_len 8192 / eager / FlashInfer off）

| 指标 | SFT step232 | RLOO step90 (n=4) | RLOO step150 (n=2) |
|---|---|---|---|
| exact reward | 21/219 = **9.59%** | **173/219 = 78.99%** | **170/219 = 77.63%** |
| terminal valid | 35/219 = 16.0% | 217/219 = 99.1% | 218/219 = 99.5% |
| tool calls（成功/失败） | 240 (240/0) | 438 (435/3) | 439 (436/3) |
| generation calls | 455 | 657 | 658 |
| zero reward | 198 | 46 | 49 |

- step90 报告：`/root/rloo-step90-native.json`，SHA256 `9ac3b533fafcd8dc18a45d5ab0d673bb631b650c7c8cd520f96c72984c5ba874`
- step150 报告：`/root/rloo-step150-native.json`，SHA256 `a2c5d6a44f8aadc6efae6494d01e6be69e6bcb9ceb2710d63115d5b629cf32e8`

**78.99% vs 77.63%：-3 条（-1.36pp），在 219 条样本的噪声范围（±~5.5% CI）内，不能判定 n=2 退化。**

---

## 3. Bad case 明细

### 3.1 分类统计

| | step90 (n=4) | step150 (n=2) |
|---|---|---|
| 总 bad（reward=0） | 46 | 49 |
| 格式错（terminal_answer_valid=false） | 2 | 1 |
| 预测错（格式对、答案错） | 44 | 48 |

### 3.2 step150 唯一格式错样本

```json
{"errors": [], "generation_calls": 4, "num_turns": 8,
 "response_tokens": 512, "reward": 0.0,
 "source_id": "f8dabf76-c644-501b-80f4-724c70c3e41c",
 "successful_tool_calls": 2, "terminal_answer_valid": false,
 "terminal_exact_match": false, "tool_call_failures": 1, "tool_calls": 3}
```

→ **response_tokens=512 触顶 + 1 次工具失败**：响应预算耗尽、无终局 JSON（`response_length=512` 截断风险实例）。

### 3.3 观察

- **两轮 bad 高度重合**：46/49 与 step90 的 bad 相同（稳定 hard cases），仅新增 3 条（含格式错 `f8dabf76`）；
- **瓶颈是分类准确率**（48/49 是格式对但答案错），不是 agent-loop 行为；
- 完整明细（含每条 num_turns/response_tokens/tool_calls/valid/errors）见 `tmp/bad-cases-2026-08-28.json`。

---

## 4. Checkpoint Manifest

| 路径 | step | 大小 | model SHA256 | eval |
|---|---|---|---|---|
| `rloo-qwen35-full-epoch1-sdpa/checkpoints/global_step_90` | 90 (n=4) | 174.7 MB | `1291bb8b20f5b722bb03f13f9f75d7e80ff34e1b87f153f8a9e8a2c014a694f4` | 78.99% |
| `rloo_n4_to_n2/checkpoints/global_step_150` | 150 (n=2) | 174.7 MB | `620ecd457b32bee45a9797f364e4770860db0c823819141e2a62d57f8860422c` | 77.63% |
| `/root/qwen35-merged-step232` | SFT merged base | 18.8 GB | — | 9.59%（SFT） |

- checkpoint 为 LoRA-only（`save_lora_only=True`），每 rank 一文件（当前 world_size=1）；
- merged 模型 `/root/autodl-tmp/qwen35-merged-rloo150` 已删除（eval 完成，可随时由 checkpoint+base 重建）；
- 完整 manifest：`tmp/checkpoint-manifest-2026-08-28.json`。

---

## 5. 暴露的问题清单

### P0 配置一致性
1. **thinking 分叉**：SFT 数据 496/496 assistant 带 `reasoning_content`（luna 生成，用户裁决），SFT 训练渲染含 `<think>`；但 RLOO 训练/评估 `enable_thinking=False`（从 smoke 继承，未按"方向 2"更新）。**当前 78.99% 是关 think 成绩**，与 SFT 数据不一致。
2. **n 混杂**：step90=n4、step150+=n2，两轮 eval 混两个变量，无法干净归因；n=2 的 RLOO baseline 方差更大。

### P1 复现性
3. **正式 RLOO 启动器不在 git**（`rloo_qwen35_formal.py` 及其 --resume/--n-rollout 扩展、`eval_rloo_checkpoint.sh` 均在服务器 run/，gitignored）；git 内 `rloo_experiment.py` 服务器版（think 关）与主线（think 开）分叉。
4. **verl site-packages patches 未逐字节比对**（patch 文件在 git，服务器应用状态未验证）。
5. **runtime-local 资产不入库**（registry/corpus/manifest），git 拉取无法直接复现。

### P2 训练信号
6. **训练 reward 饱和**：step50+ pg_loss≈0、梯度由 KL（0.05–0.13）驱动；entropy 0.02–0.03 低熵——训练 reward 1.0 不可作泛化证据。
7. **训练/泛化差距大**：train reward ~1.0 vs held-out 77–79%。
8. **response_length=512 截断**：bad case `f8dabf76` 实测触顶无终局；开 thinking 后 think 块将进一步挤占预算。

### P3 数据/评估
9. **219 条测试集偏小**（CI ±~5.5%），77.63 vs 78.99 不显著；未做 F1/更大集对比。
10. **eval 未持久化 raw trajectory**：报告只有统计字段，bad case 的原始生成文本不可得（需重跑 eval 并保存）。

---

## 6. Provenance

- 服务器：AutoDL，RTX PRO 6000 96GB，venv `verl-qwen35`（VeRL 0.9.0 / vLLM 0.27.1 / transformers 5.10.4 / torch 2.13.0+cu130）；
- 基模：`/root/autodl-tmp/models/Qwen3.5-9B`；RLOO base：`/root/qwen35-merged-step232`；
- 训练数据：`/root/autodl-tmp/rloo-smoke/rl/{train 14521, val 1926}`；评估集：`release-b/test.parquet`（219，与 train/val 零重叠，已核）；
- 日志：`rloo-to-300.log`（n2 段）、`rloo-qwen35-full-epoch1-sdpa-resume90b.log`（n4 段）、`rloo-n4-to-n2.log`；
- 提取工具：`tmp/verify_chat_template_tooluse.py`（待运行）、`tmp/bad-cases-2026-08-28.json`、`tmp/checkpoint-manifest-2026-08-28.json`。
