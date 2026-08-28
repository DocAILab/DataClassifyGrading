# verl-0.9.0 AmberFalcon runtime patches (tracked bundle, plan A)

Source: live server venv `<SERVER_VENV>` vs official wheel
`verl-0.9.0-py3-none-any.whl` (forensics: full 420-file byte comparison;
416 identical / 4 patched / 0 missing; evidence
`tmp/phase-a1-cfgdump/verl-patches.diff.txt`,
`tmp/phase-a1-sync-evidence-2026-08-27/11_verl_patches_forensics.txt`).

This bundle is replayable from the official wheel: applying all four patches
produces the fixed target sha256 values recorded below (verified locally,
2026-08-27, against wheel `verl-0.9.0-py3-none-any.whl` 1,616,124 B). The
live server venv retains the pre-fix hashes until an explicitly managed runtime
rollout; CPU validation uses a temporary module overlay and never mutates it.

## Patch inventory

| patch | target (relative to site-packages) | applied sha256 (installed) | source mtime (unix) | lines | status |
|---|---|---|---|---|---|
| `agent_loop-debug.patch` | `verl/experimental/agent_loop/agent_loop.py` | `902cc8c4007b944974d77c54bc1ce227df49de4390e5b3a0fc831f5cf0a4a801` | 1787775452.5897617 | +15 | **DEBUG ONLY — optional, P2 cleanup; do not apply for formal runs** |
| `chat_template-system-first.patch` | `verl/utils/tokenizer/chat_template.py` | `58031af7a001a1208129b271f110e9cf94a3978874fadc5da43db9eec0322578` | 1787758027.2868876 | +16 | required |
| `multiturn_sft_dataset-prefix-diff-answer-mask.patch` | `verl/utils/dataset/multiturn_sft_dataset.py` | `ce7486288a68a85a0777d9e587688501e09603533703e57d91e4f2c85139ecd9` | bundle fix | +225 | required |
| `losses-scheme-c.patch` | `verl/workers/utils/losses.py` | `f107371e5c77b8f81800d3676d85894a64ad6646ea78f83ecb59d768ebc09a5c` | bundle fix | +18 | required |

## Patch file sha256 (this bundle)

| file | sha256 |
|---|---|
| `agent_loop-debug.patch` | `b257a87a7c6aaa2ddbfba94fc4efefbfb860e5679eda4410ca6ba71a6cdb61ed` |
| `chat_template-system-first.patch` | `1e493510dc41ddf455587555d1f223d2c3479e570ca94d7ecf6ac91db1975d36` |
| `multiturn_sft_dataset-prefix-diff-answer-mask.patch` | `245165c9117b14ab5acfd5255b02dd8ea323e1f1d2b2c6bcc176cfb00054e4ad` |
| `losses-scheme-c.patch` | `6463eab23787286907f3ffbd5dc9850e26fa241b8b75e84322cfb32d4d633829` |

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
   - **answer_mask channel** (fail-closed): only the terminal assistant's
     rendered `content` is eligible.  The content boundary is computed by
     rendering the prefix with the terminal content emptied and taking the
     **unique** split of the two deltas (`real == empty[:i] + content +
     empty[i:]`); when that render is unavailable, the complete content token
     sequence is located only if it occurs **exactly once** in the assistant
     delta.  Any ambiguity (e.g. the identical JSON repeated verbatim inside
     think) fails closed to an all-zero answer_mask — answer/level values are
     never searched for in reasoning or the full input.  Value spans inside
     the content region use tokenizer offsets when available, else a
     content-local token-subsequence lookup that also fails closed on
     truncation/encoding mismatch.  The mask is carried through
     cat/padding/truncation.
4. **`losses-scheme-c.patch`** — SFT loss upweights the answer value span
   (`w_ans=8.0`, substitution not addition) while intersecting it with the
   policy loss mask so tool observations cannot contribute:
   `loss = -(w_ans·masked_sum(log_prob, loss_mask·ans_mask) +
   masked_sum(log_prob, loss_mask·(1−ans_mask))) / batch_num_tokens · dp_size`,
   with `torch.roll(-1)` next-token alignment (same as loss_mask) and fallback
   to the vanilla loss when `answer_mask` is absent.

## Scheme C semantics vs this project's contracts

- Terminal strict JSON keys are exactly `{"answer", "level"}` (contract:
  `src/agent/training/rl/native_tools.py::parse_final_tool_answer`, key-set
  check `set(value) != {"answer","level"}`; prompts: stage-2 system message).
  **The patch regex targets exactly these keys — no adaptation needed.**
- Scope note: the 8× weight covers only the **answer/level value spans**, not
  the whole terminal JSON; JSON framework/think/format tokens stay at 1×.
  Stage-1 `candidates` output has no `answer`/`level` keys → no upweight
  (intended: stage-1 is not the final answer).
- Edge cases (documented, see think proposal §6 and the fail-closed contract):
  the terminal content anchor refuses to mark values when the empty-content
  boundary is ambiguous or the complete content sequence is not uniquely
  present; it never falls back to searching reasoning or the full input.
  Prefix-diff exactness with `tools=` set is not yet verified (patch was
  verified on `[user] → [user,assistant]` without tools) — validate on real
  tool-trajectory SFT data.

## Usage

```bash
# fresh venv replay (apply required patches; optional DEBUG excluded by default)
bash script/verl/common/patches/apply_verl_patches.sh \
  <SERVER_VENV>/lib/python3.12/site-packages

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
