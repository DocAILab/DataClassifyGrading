"""Copy a clean pinned local research runtime, without network or data contents."""
import argparse
from pathlib import Path
import shutil
import subprocess
from artifacts import ROOT, REVISION, TEAM_COMMIT, put, read, sha


def vendor(source, download_manifest):
    source = Path(source)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--', 'research/finance_program_v1'], cwd=source, text=True)
    if commit != TEAM_COMMIT or dirty:
        raise ValueError('source must be the clean pinned team research commit')
    research = source / 'research/finance_program_v1'
    target = ROOT / 'upstream'
    target.mkdir(exist_ok=False)
    for path in sorted((research / 'evaluation/runtime').rglob('*')):
        if path.is_file() and path.suffix in ('.py', '.md', '.json') and '__pycache__' not in path.parts:
            dest = target / 'runtime' / path.relative_to(research / 'evaluation/runtime')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    for name in ('local_smoke.py', 'download_model.py'):
        dest = target / 'backend' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(research / 'evaluation/legacy' / name, dest)
    put(ROOT / 'upstream.json', {'repository': 'https://github.com/DocAILab/DataClassifyGrading',
                                'commit': TEAM_COMMIT, 'runtime_origin': 'research/finance_program_v1/evaluation/runtime',
                                'files': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(target.rglob('*')) if p.is_file()}})
    manifest = read(download_manifest)
    if manifest['repository'] != 'DocAILab/DCG' or manifest['revision'] != REVISION:
        raise ValueError('download manifest revision differs')
    names = {f'DCG_VEHICLE/DCG_VEHICLE_{s}.json' for s in ('SFT', 'TRAIN', 'VAL', 'TEST', 'CORPUS')}
    hashes = {r['path']: r['sha256'] for r in manifest['files'] if r['path'] in names}
    if set(hashes) != names:
        raise ValueError('missing pinned source files')
    put(ROOT / 'pinned-inputs.json', {'repository': manifest['repository'], 'revision': REVISION,
                                    'download_manifest_sha256': sha(download_manifest), 'files': hashes})
    print('Pinned runtime and five input hashes saved; no dataset contents copied.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--download-manifest', type=Path, required=True)
    a = p.parse_args()
    vendor(a.source, a.download_manifest)
