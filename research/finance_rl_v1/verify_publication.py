"""Check publication integrity and run the portable original CPU contract tests."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
TRAIN = ROOT/'experiments/finance_program_grpo_full_migrated_20261027_r3'
EVAL = ROOT/'experiments/finance_program_grpo_val422_20261028_r4'

def main():
    manifest = json.loads((ROOT/'publication-manifest.json').read_bytes())
    expected = {r['path']: r for r in manifest['files']}
    actual = {p.relative_to(ROOT).as_posix(): p for p in ROOT.rglob('*')
              if p.is_file() and p.name != 'publication-manifest.json'
              and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts}
    if actual.keys() != expected.keys():
        raise ValueError('publication file set differs from manifest')
    for name, p in actual.items():
        data = p.read_bytes()
        if len(data) != expected[name]['bytes'] or hashlib.sha256(data).hexdigest() != expected[name]['sha256']:
            raise ValueError('publication identity changed: '+name)
        if p.suffix == '.py':
            ast.parse(data.decode('utf-8'), filename=name)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    train_tests = [p.stem for p in sorted((TRAIN/'tests').glob('test_*.py'))]
    code = "import sys, unittest, torch; sys.path.insert(0, 'tests'); "
    code += 'suite=unittest.defaultTestLoader.loadTestsFromNames('+repr(train_tests)+'); '
    code += 'result=unittest.TextTestRunner(verbosity=2).run(suite); '
    code += 'assert not torch.cuda.is_initialized(); sys.exit(not result.wasSuccessful())'
    subprocess.run([sys.executable, '-B', '-c', code], cwd=TRAIN, env=env, check=True)
    subprocess.run([sys.executable, '-B', '-m', 'unittest', '-v',
                    'test_candidate_contract', 'test_process_contract'], cwd=EVAL, env=env, check=True)
    print(json.dumps(dict(passed=True, files=len(actual),
                          training_test_modules=len(train_tests), evaluation_test_modules=2,
                          CUDA_VISIBLE_DEVICES='-1', full_model_inference=False,
                          historical_native_replays_repeated=False)))

if __name__ == '__main__':
    main()
