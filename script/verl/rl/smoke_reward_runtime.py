"""Phase 12 step-1 smoke: reward RUNTIME through the real VeRL reward adapter.

Drives the exact entrypoint that will be wired into VeRL
(``agent.training.rl.verl_adapter.compute_score``) against REAL Phase-8 RL
parquet rows. For each sampled Stage-1 / Stage-2 row it synthesizes three
completions and asserts the returned reward matches the frozen RewardConfig:

- ``correct``        -> expected FULL_REWARD (1.0)
- ``valid but wrong``-> expected stage1_valid_miss (0.3) / stage2_partial (0.5)
- ``malformed``      -> expected INVALID_REWARD 0.0, never raising

No parser / reward table is reimplemented: scores come from the adapter
(and therefore the shared choice parser + canonical decode + reward table).
Requires only the task layer (no verl, no torch), so it runs on CPU in any
venv.

Usage (from repo root, with a real RL parquet dir):
  python -m script.verl.rl.smoke_reward_runtime \
    --dataset-dir data/rl/pers_info \
    --dataset pers_info \
    --registry cfg/task/registry/pers_info.registry.json \
    --num-samples 4 --report tmp/phase12/reward_runtime_smoke.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pyarrow.parquet as pq

from agent.task.contracts import LeafRegistry
from agent.task.prompt_choices import (
    PromptChoiceRegistry,
    encode_stage2_answer,
)
from agent.training.rl.verl_adapter import compute_score, parse_data_source

FULL = 1.0
INVALID = 0.0
STAGE1_MISS = 0.3
STAGE2_PARTIAL = 0.5


def _stage1_completions(choices: PromptChoiceRegistry, gt: str) -> dict[str, tuple[str, float]]:
    all_ids = choices.choice_ids
    gt_id = choices.choice_id_of(gt)
    others = [cid for cid in all_ids if cid != gt_id]
    correct_ids = [gt_id] + others[:4]
    wrong_ids = others[:5]
    return {
        "correct": (
            json.dumps({"candidates": correct_ids}, ensure_ascii=False),
            FULL,
        ),
        "validwrong": (
            json.dumps({"candidates": wrong_ids}, ensure_ascii=False),
            STAGE1_MISS,
        ),
        "malformed": ("this is not a json completion", INVALID),
    }


def _stage2_completions(gt: str, candidates: list[str]) -> dict[str, tuple[str, float]]:
    canonical_answer = None
    if gt in candidates:
        canonical_answer = encode_stage2_answer(gt, candidates)
    # a wrong valid answer: position of a candidate that is not the GT
    gt_pos = candidates.index(gt) if gt in candidates else None
    wrong_idx = (gt_pos + 1) % 5 if gt_pos is not None else 0
    wrong_answer = encode_stage2_answer(candidates[wrong_idx], candidates)
    return {
        "correct": (
            json.dumps({"answer": canonical_answer}, ensure_ascii=False),
            FULL if canonical_answer is not None else None,
        ),
        "validwrong": (
            json.dumps({"answer": wrong_answer}, ensure_ascii=False),
            STAGE2_PARTIAL,
        ),
        "malformed": (json.dumps({"answer": "9"}, ensure_ascii=False), INVALID),
        "strict_invalid": ("not json", INVALID),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--report")
    args = parser.parse_args(argv)

    registry = LeafRegistry.from_path(args.registry)
    choices = PromptChoiceRegistry.from_registry(registry)
    rows = pq.read_table(Path(args.dataset_dir) / "train.parquet").to_pylist()
    stage1_rows = [r for r in rows if r["data_source"].endswith("stage1")][: args.num_samples]
    stage2_rows = [r for r in rows if r["data_source"].endswith("stage2")][: args.num_samples]
    if not stage1_rows or not stage2_rows:
        print("error: need at least one stage1 and one stage2 row", file=sys.stderr)
        return 2

    cases: list[dict] = []
    failures: list[str] = []

    def run_case(
        stage: str,
        row: dict,
        label: str,
        solution: str,
        expected: float | None,
    ) -> None:
        extra_info = row["extra_info"]
        actual = compute_score(
            row["data_source"],
            solution,
            row["reward_model"]["ground_truth"],
            extra_info,
        )
        ok = expected is not None and abs(actual - expected) < 1e-9
        cases.append(
            {
                "case": label,
                "stage": stage,
                "data_source": row["data_source"],
                "source_id": extra_info.get("source_id"),
                "ground_truth": row["reward_model"]["ground_truth"],
                "solution": solution,
                "expected": expected,
                "actual": actual,
                "pass": ok,
            }
        )
        if not ok:
            failures.append(
                f"{row['data_source']}/{label}: expected {expected}, got {actual}"
            )

    for row in stage1_rows:
        ds, stage = parse_data_source(row["data_source"])
        assert stage == "stage1" and ds == args.dataset
        for label, (solution, expected) in _stage1_completions(choices, row["reward_model"]["ground_truth"]).items():
            run_case("stage1", row, label, solution, expected)

    for row in stage2_rows:
        candidates = row["extra_info"].get("candidates")
        for label, (solution, expected) in _stage2_completions(
            row["reward_model"]["ground_truth"], candidates
        ).items():
            run_case("stage2", row, label, solution, expected)

    summary = {
        "dataset": args.dataset,
        "num_stage1_rows": len(stage1_rows),
        "num_stage2_rows": len(stage2_rows),
        "cases": cases,
        "pass": not failures,
        "failures": failures,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
