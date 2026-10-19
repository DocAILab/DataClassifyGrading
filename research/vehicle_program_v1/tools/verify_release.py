"""Verify public artifact bytes and completion counts; no model or private data."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'results/files.sha256.json').read_text(encoding='utf-8'))
    for relative,expected in manifest.items():
        path=(root/relative).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise ValueError('publication file identity differs: '+relative)
    rows=[json.loads(line) for line in (root/'trajectories/val397.redacted.jsonl').read_text(encoding='utf-8').splitlines()]
    summary=json.loads((root/'results/evidence-summary.json').read_text(encoding='utf-8'))
    assert [row['public_id'] for row in rows]==[f'val-{i:04d}' for i in range(1,398)]
    assert sum(row['terminated'] for row in rows)==summary['metrics']['terminated']==115
    assert sum(row['receipt_status']=='backend_error' for row in rows)==17
    return {'pass':True,'verified_files':len(manifest),'public_rows':len(rows),
            'independent_semantic_accuracy_verified_from_public_rows':False}


if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1])))
