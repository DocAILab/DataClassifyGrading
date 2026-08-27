# verl-0.9.0 AmberFalcon runtime patches (tracked bundle, plan A)

Source: live server venv `/root/autodl-tmp/envs/verl-qwen35` vs official wheel
`verl-0.9.0-py3-none-any.whl` (forensics: full 420-file byte comparison;
416 identical / 4 patched / 0 missing; evidence
`tmp/phase-a1-cfgdump/verl-patches.diff.txt`,
`tmp/phase-a1-sync-evidence-2026-08-27/11_verl_patches_forensics.txt`).

This bundle reproduces the server-installed files byte-for-byte: applying all
four patches to the official wheel produces the exact installed sha256 values
(verified locally, 2026-08-27, against wheel
`verl-0.9.0-py3-none-any.whl` 1,616,124 B).

## Patch inventory

| patch | target (relative to site-packages) | applied sha256 (installed) | source mtime (unix) | lines | status |
|---|---|---|---|---|---|
| `agent_loop-debug.patch` | `verl/experimental/agent_loop/agent_loop.py` | `902cc8c4007b944974d77c54bc1ce227df49de4390e5b3a0fc831f5cf0a4a801` | 1787775452.5897617 | +15 | **DEBUG ONLY — optional, P2 cleanup; do not apply for formal runs** |
| `chat_template-system-first.patch` | `verl/utils/tokenizer/chat_template.py` | `58031af7a001a1208129b271f110e9cf94a3978874fadc5da43db9eec0322578` | 1787758027.2868876 | +16 | required |
| `multiturn_sft_dataset-prefix-diff-answer-mask.patch` | `verl/utils/dataset/multiturn_sft_dataset.py` | `060f0adfe4caf5a4a19b0b7dd443ec417117070db16039ee8905691bd8aa85ba` | 1787766113.2599778 | +134 | required |
| `losses-scheme-c.patch` | `verl/workers/utils/losses.py` | `7035e59a0678b8172c608f72d296ef9893babf35fa27650f72be8d54d1d6fdca` | 1787766087.019359 | +21 | required |

## Patch file sha256 (this bundle)

| file | sha256 |
|---|---|
| `agent_loop-debug.patch` | `b257a87a7c6aaa2ddbfba94fc4efefbfb860e5679eda4410ca6ba71a6cdb61ed` |
| `chat_template-system-first.patch` | `1e493510dc41ddf455587555d1f223d2c3479e570ca94d7ecf6ac91db1975d36` |
| `multiturn_sft_dataset-prefix-diff-answer-mask.patch` | `683d49068c5ad4ad9f4e428fd712b6aecdadc6535528500327e5576bef0603bb` |
| `losses-scheme-c.patch` | `eb9588ce620c722109fca0df3298b138f922744be2dd7c46b94cfe753feeedc4` |

## Functional description (per patch)

1. **`agent_loop-debug.patch`** — three `print(f"[DEBUG] ...", flush=True)` lines
   in agent-loop loading. No functional change. Marked optional: **do not apply
   for formal runs** (debug noise); P2 cleanup alongside CRLF/debug-redaction
   follow-ups.
2. **`chat_template-system-first.patch`** — Qwen3.5 chat template requires the
   leading system message to stay first; VeRL's internal `dummy_user_message`
   stitching previously prepended before the whole list. Patch inserts the
   dummy user after the system message instead.
3. **`multiturn_sft_dataset-prefix-diff-answer-mask.patch`** —
   - Qwen3.5 templates cannot render standalone messages (thinking structure +
     system-first constraint): render the full prefix up to the current
     message and diff against the previous prefix (`prefix-diff`); replaces the
     legacy system-prompt strip.
   - Per-row `enable_thinking` plumbing retained.
   - **answer_mask channel**: for each assistant message, locate the
     `"answer"` / `"level"` value spans in `content` (regex
     `"answer"\s*:\s*"([^"]+)"` / `"level"\s*:\s*"([^"]+)"`), encode the value,
     find the token-id subsequence in `input_ids` (first match wins), set
     `answer_mask=1` on that span; carried through cat/padding/truncation.
4. **`losses-scheme-c.patch`** — SFT loss upweights the answer value span
   (`w_ans=8.0`, substitution not addition):
   `loss = -(w_ans·masked_sum(log_prob, ans_mask) + masked_sum(log_prob,
   loss_mask·(1−ans_mask))) / batch_num_tokens · dp_size`, with
   `torch.roll(-1)` next-token alignment (same as loss_mask) and fallback to
   the vanilla loss when `answer_mask` is absent.

## Scheme C semantics vs this project's contracts

- Terminal strict JSON keys are exactly `{"answer", "level"}` (contract:
  `src/agent/training/rl/native_tools.py::parse_final_tool_answer`, key-set
  check `set(value) != {"answer","level"}`; prompts: stage-2 system message).
  **The patch regex targets exactly these keys — no adaptation needed.**
- Scope note: the 8× weight covers only the **answer/level value spans**, not
  the whole terminal JSON; JSON framework/think/format tokens stay at 1×.
  Stage-1 `candidates` output has no `answer`/`level` keys → no upweight
  (intended: stage-1 is not the final answer).
- Edge cases (documented, see think proposal §6): content-string search can
  mis-mark a same-shape `"answer":"..."` string inside think text (rare;
  first-match break); prefix-diff exactness with `tools=` set is not yet
  verified (patch was verified on `[user] → [user,assistant]` without tools)
  — validate on real tool-trajectory SFT data.

## Usage

```bash
# fresh venv replay (apply required patches; optional DEBUG excluded by default)
bash script/verl/common/patches/apply_verl_patches.sh \
  /root/autodl-tmp/envs/<venv>/lib/python3.12/site-packages

# include the DEBUG patch (not recommended for formal runs)
INCLUDE_DEBUG=1 bash script/verl/common/patches/apply_verl_patches.sh <site-packages>

# read-only verification of an existing install
bash script/verl/common/patches/verify_verl_patches.sh <site-packages>
```

Idempotency: applying to an already-patched tree is detected via target
sha256 and skipped (no double-apply). On failure, already-applied patches are
reverted with `patch -R` (see script header).

## Key-name gap check (coordinator request)

`answer`/`level` keys match the terminal contract exactly → **no gap, no patch
adaptation, no contract change** (reported to coordinator; only the scope
notes above apply).
