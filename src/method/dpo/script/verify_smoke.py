"""Verify a one-update DPO smoke run before full training."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from method.dpo.smoke_verification import verify_smoke


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--verification-dir", required=True)
    args = parser.parse_args(argv)
    report = verify_smoke(args.training_dir, args.verification_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
