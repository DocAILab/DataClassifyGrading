"""Immutable complete post-stage evidence snapshot; excludes external base weights."""
from pathlib import Path
import tarfile
from runtime_io import digest, put, read, verify_manifest


def main():
    root = Path(__file__).resolve().parent
    verify_manifest(root)
    read(root / 'runs/pipeline.exit.json')
    out = root / 'exports/final-01'
    out.mkdir(parents=True, exist_ok=False)
    files = sorted(p for p in root.rglob('*') if p.is_file()
        and 'exports' not in p.relative_to(root).parts and '__pycache__' not in p.parts
        and p.relative_to(root).as_posix() != 'runs/pipeline.log')
    # pipeline.log is still open in the outer supervisor; stage logs/true exits are complete.
    items = [{'path': p.relative_to(root).as_posix(), 'sha256': digest(p), 'bytes': p.stat().st_size} for p in files]
    put(out / 'members.json', items)
    archive = out / 'student-sft200-evidence.tar.gz'
    with archive.open('xb') as stream, tarfile.open(fileobj=stream, mode='w:gz') as tar:
        for p in files:
            tar.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
        tar.add(out / 'members.json', arcname='archive-members.json', recursive=False)
    put(out / 'archive.json', {'archive_sha256': digest(archive), 'bytes': archive.stat().st_size,
        'members': len(items) + 1, 'member_manifest_sha256': digest(out / 'members.json'),
        'pipeline_exit': read(root / 'runs/pipeline.exit.json'),
        'supervisor_log_excluded_because_writer_open': True})


if __name__ == '__main__':
    main()
