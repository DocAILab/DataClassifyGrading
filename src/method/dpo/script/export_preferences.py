"""Export mined train-only scores as conversational DPO preferences."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from method.dpo.preference_data import export_preferences


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--score-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    report = export_preferences(
        args.input_dir,
        args.output_dir,
        args.registry,
        args.task_config,
        args.score_path,
        seed=args.seed,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
