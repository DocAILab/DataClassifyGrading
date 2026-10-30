"""Reuse original locks/absolute deadline, additionally pin reused r1 source."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'finance_program_skill_interface_smoke_20261021_r1'
spec = importlib.util.spec_from_file_location('previous_smoke_launcher', PREVIOUS / 'launch_smoke.py')
launcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(launcher)
launcher.ROOT = ROOT
original_verify = launcher.verify


def verify():
    original_verify()
    for row in json.loads((ROOT / 'freeze.json').read_bytes())['dependency_files']:
        file = ROOT.parent / row['path']
        if file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('reused source dependency changed: ' + row['path'])


launcher.verify = verify


if __name__ == '__main__':
    raise SystemExit(launcher.watch() if sys.argv[1:] == ['--watch'] else launcher.main())
