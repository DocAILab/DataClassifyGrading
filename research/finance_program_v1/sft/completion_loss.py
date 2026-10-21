"""Full-context causal loss using only logits needed for assistant supervision."""
# 输入上下文保持完整，仅保留计算响应损失所需的末尾 logits，以减少大词表输出占用的显存。

import torch
import torch.nn.functional as F


def completion_cross_entropy(suffix_logits: torch.Tensor, labels: torch.Tensor):
    if labels.ndim != 2 or suffix_logits.ndim != 3 or suffix_logits.shape[0] != labels.shape[0]:
        raise ValueError("unexpected labels/logits shapes")
    supervised = labels.ne(-100)
    counts = supervised.sum(dim=1)
    if counts.numel() != 1 or int(counts[0]) < 1:
        raise ValueError("one example with assistant supervision is required")
    count = int(counts[0])
    start = labels.shape[1] - count
    if start < 1 or not torch.all(labels[:, :start].eq(-100)) or not torch.all(supervised[:, start:]):
        raise ValueError("supervision must be one contiguous suffix after prompt")
    if suffix_logits.shape[1] != count + 1:
        raise ValueError("logits must include prompt boundary and one logit per target")
    # 因果预测向右错一位：prompt 最后位置预测首个响应 token，最后一个 logit 无对应目标。
    return F.cross_entropy(suffix_logits[:, :-1, :].float().reshape(-1, suffix_logits.shape[-1]),
                           labels[:, start:].reshape(-1))
