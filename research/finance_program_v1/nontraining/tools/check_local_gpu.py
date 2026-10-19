"""Short local CUDA arithmetic check. NOT a Planner/LLM inference benchmark."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys


def cuda_arithmetic():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('PyTorch has no usable CUDA device')
    torch.manual_seed(20261010)
    torch.backends.cuda.matmul.allow_tf32 = False
    left = torch.randn(256, 256, dtype=torch.float32)
    right = torch.randn(256, 256, dtype=torch.float32)
    cpu = left @ right
    gpu = (left.to('cuda:0') @ right.to('cuda:0')).cpu()
    torch.cuda.synchronize(0)
    error = float((cpu - gpu).abs().max())
    torch.testing.assert_close(gpu, cpu, rtol=1e-4, atol=1e-4)
    return {'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
            'device': torch.cuda.get_device_name(0), 'device_count': torch.cuda.device_count(),
            'shape': [256, 256], 'dtype': 'float32', 'max_abs_error': error,
            'passed': True, 'test_kind': 'CUDA arithmetic, not LLM inference'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New report path; existing files are rejected')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--child', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        print(json.dumps(cuda_arithmetic()))
        return 0
    if args.timeout < 1:
        parser.error('--timeout must be positive')
    if args.output is not None and args.output.exists():
        parser.error('output exists; use a new path')
    report = {'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.executable,
              'language_model_tested': False}
    try:
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--child'],
                               capture_output=True, text=True, timeout=args.timeout)
        report.update({'child_exit': child.returncode, 'stderr': child.stderr})
        if child.returncode:
            report.update({'passed': False, 'stdout': child.stdout})
        else:
            report.update(json.loads(child.stdout))
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        report.update({'passed': False, 'error': str(exc)})
    encoded = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(encoded)
    print(encoded, end='')
    return 0 if report.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
