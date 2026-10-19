"""Local immutable artifacts; no model, network or label access on import."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
REVISION = '84a9a462dcb828332f980ef121f40e1281ce1794'
TEAM_COMMIT = '9473b2c2a330b7f48dc0c1831bda11e36d428379'
MODEL_REVISION = 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
BUDGET = dict(max_programs=3, max_program_steps=10, max_skill_steps=12,
              max_planner_attempts=4, max_identical_calls=1)
GENERATION = dict(seed=20260928, context_cap=32768, max_new_tokens=512,
                  enable_thinking=False, do_sample=False, dtype='float16')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    allow_nan=False).encode('utf-8')).hexdigest()


def require_private_output(path):
    resolved = Path(path).resolve()
    roots = [ROOT] + [parent for parent in ROOT.parents if (parent / '.git').exists()]
    if any(resolved.is_relative_to(root) for root in roots):
        raise ValueError('private data/artifacts must stay outside the code and Git repository')


def put(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    # A brief status reader on Windows can deny replace. Retry the file commit,
    # never the HTTP request that produced the evidence.
    for attempt in range(101):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            if attempt == 100:
                raise
            time.sleep(.01)


def code_inventory():
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(ROOT.rglob('*'))
            if p.is_file() and p.suffix in ('.py', '.json', '.md', '.ps1')
            and '__pycache__' not in p.parts}


def verify_vendor():
    source = read(ROOT / 'upstream.json')
    if source['commit'] != TEAM_COMMIT:
        raise ValueError('team commit differs')
    actual = {p.relative_to(ROOT).as_posix(): sha(p)
              for p in sorted((ROOT / 'upstream').rglob('*')) if p.is_file()
              and '__pycache__' not in p.parts}
    if actual != source['files']:
        raise ValueError('vendored runtime changed')


def load_study(package):
    package = Path(package)
    manifest = read(package / 'manifest.json')
    if manifest['code'] != code_inventory():
        raise ValueError('experiment code changed; prepare a new package')
    verify_vendor()
    if sha(package / 'inference/study.json') != manifest['study_sha256']:
        raise ValueError('inference study identity changed')
    study = read(package / 'inference/study.json')
    if study['budget'] != BUDGET or study['skill_pool'] != 'F66' or study['generation'] != GENERATION:
        raise ValueError('frozen F66 budget differs')
    ids = [r['id'] for r in study['rows']]
    if ids != study['expected_ids'] or len(ids) != len(set(ids)) or not ids:
        raise ValueError('study identity/order differs')
    for row in study['rows']:
        if set(row) != {'id', 'domain', 'metadata'} or not set(row['metadata']) <= {'field_name', 'field_description'}:
            raise ValueError('study metadata allowlist violated')
    return study, manifest


def load_controller():
    import sys
    runtime = ROOT / 'upstream/runtime'
    sys.path.insert(0, str(runtime))
    import program_loop
    if not Path(program_loop.__file__).resolve().is_relative_to(runtime):
        raise ValueError('a different program_loop is already imported')
    return program_loop.build_program_controller
