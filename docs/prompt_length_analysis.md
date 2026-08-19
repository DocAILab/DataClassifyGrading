# Prompt Length Analysis (Phase 9)

Recomputed on the **Phase 8 real SFT/RL prompts** (choice protocol) against the
legacy canonical-id baseline, using the `Qwen/Qwen2.5-7B-Instruct` tokenizer
(snapshot `a09a3545`, tokenizer.json only). Measurement = token count of the
full chat-template-rendered conversation (identical semantics to the legacy
`prompt_stats.py` baseline; p95 uses the legacy definition
`sorted[int(0.95·n)-1]`). Old stats are kept untouched as the legacy baseline.

Generated stats: `artifacts/generated/prompt_stats/` (canonical_id_baseline/ =
old canonical-id stats, choice_id/ = new choice-protocol stats).

## 1. Old vs new p95 / max (train split, tokens)

### p95
| dataset | stage | old p95 | new p95 | reduction % |
|---|---|---|---|---|
| finance | stage1 | 5950 | 2543 | **57.3%** |
| finance | stage2 | 603 | 560 | 7.1% |
| infra | stage1 | 5073 | 2587 | 49.0% |
| infra | stage2 | 347 | 315 | 9.2% |
| pers_info | stage1 | 517 | 260 | 49.7% |
| pers_info | stage2 | 232 | 196 | 15.5% |
| shougang | stage1 | 5073 | 2588 | 49.0% |
| shougang | stage2 | 345 | 313 | 9.3% |

### max
| dataset | stage | old max | new max | reduction % |
|---|---|---|---|---|
| finance | stage1 | 5959 | 2553 | **57.2%** |
| finance | stage2 | 794 | 740 | 6.8% |
| infra | stage1 | 5074 | 2588 | 49.0% |
| infra | stage2 | 348 | 316 | 9.2% |
| pers_info | stage1 | 523 | 266 | 49.1% |
| pers_info | stage2 | 236 | 202 | 14.4% |
| shougang | stage1 | 5157 | 2672 | 48.2% |
| shougang | stage2 | 408 | 376 | 7.8% |

## 2. Stage 1 is the main token-saving source

Per-record combined mean (train): finance 6483.7 → 3029.4 tokens (**−53.3%**,
abs. saving ≈ 3454); infra/shougang 5393 → 2876 (**−46.7%**); pers_info 737.9 →
446.8 (−39.5%). ~99% of the absolute saving comes from **Stage 1**
(finance/infra/shougang ≈ 98–99%), where the full 233-category registry used to
be rendered with verbose canonical ids and is now a compact
`[choice_id, display_name]` catalog. Stage 2 (the 5-candidate bundle) was
already compact, so it gains only 7–15%.

## 3. finance vs shougang: which is longest now — and a reversal

- **shougang** is now the longest (global max 2689, test stage1; prompt-only
  2670) vs finance (max 2553).
- This **reverses the old baseline** (finance 5959 > shougang 5157). Reasons:
  finance's old ids were long (`finance:客户.个人….`) so choice ids save 57%;
  shougang's old ids were short (`A1-1-1`) so they save 49%. The remaining
  longest rows are driven by **long metadata text** in the data (shougang's
  longest row carries a 17-item enumerated `field_description`; finance's
  longest row has tiny metadata), not by the registry.

## 4. Worst case vs context

Global worst case = **2689 tokens** (shougang test, Stage 1) ≈ **8.2% of the
32768 context** (prompt-only 2670 ≈ 8.15%). The new prompts are safely below
the context limit with a large margin — the old worst case was ~6000 tokens
(~18%), so the choice protocol roughly halves the headroom consumed.

## 5. Baselines & reproducibility

`artifacts/generated/prompt_stats/canonical_id_baseline/` keeps the old
canonical-id stats (`prompt_token_stats_*.json`, `prompt_stats_*.json`,
`prompt_token_stats_summary.json`) unchanged for regressions and comparison;
`artifacts/generated/prompt_stats/choice_id/` holds the new choice-protocol
stats. Regenerate with
`python -m script.verl.sft.prompt_stats --dataset-dir data/sft/<ds> --model
Qwen/Qwen2.5-7B-Instruct --report <out>.json` (or the Phase-9 script under
`tmp/re_export/` reproducing the exact tables above).

## 6. Note: PromptChoice is not only about length

The choice protocol exists primarily to **decouple the internal canonical
`category_id` from model actions** — ids must never be generated as actions
(fragile, leaks taxonomy). Token savings (≈50% Stage 1) are a welcome side
effect. See `docs/design/prompt_interface.md` and
`docs/design/data_contract.md`.
