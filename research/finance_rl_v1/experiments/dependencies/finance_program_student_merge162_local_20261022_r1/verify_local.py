"""Independent offline byte/partition verification; no preparation helpers imported."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
merge = root.parent / 'finance_program_teacher_merge_20261022_r1'
evalroot = root.parent / 'finance_program_student_r9_eval80_20261022_r1'
report = json.loads((root / 'reviews/local-preparation-01.json').read_bytes())
checked = []
for ref in report['inputs'] + json.loads((merge / 'reviews/final-delivery-seal-01.json').read_bytes())['files']:
    path = Path(ref['path'])
    raw = path.read_bytes()
    assert len(raw) == ref['bytes'] and hashlib.sha256(raw).hexdigest() == ref['sha256'], str(path)
    checked.append(str(path))
original = [json.loads(line) for line in (merge / 'data/planner-curated.jsonl').read_bytes().splitlines()]
index = {r['id']: r for r in json.loads((merge / 'reviews/curation-index-01.json').read_bytes())['rows'] if r.get('exported')}
evaluation = json.loads((evalroot / 'selection.json').read_bytes())['selected']
groups = {r['group_sha256'] for r in evaluation}
sources = {r['source_id'] for r in evaluation}
expected = [r for r in original if index[r['id']]['group'] not in groups and index[r['id']]['source_id'] not in sources]
assert len(original) == 162 and len(expected) == 90
for name, wanted in [('prepared-diagnostic162', original), ('prepared-group-isolated', expected)]:
    path = root / name / 'train-candidates.jsonl'
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    assert rows == wanted, name
    assert all(set(r) == {'id', 'messages', 'completion'} for r in rows)
    manifest = json.loads((root / name / 'preparation.json').read_bytes())
    assert manifest['candidates'] == len(wanted) and manifest['train_ready'] is False
    assert manifest['candidate_file']['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
assert report['current_eval_overlap_examples'] == 72 and report['current_eval_overlap_groups'] == 48
assert report['tokenizer_checked'] is False and report['training_started'] is False
files = []
for path in sorted(root.rglob('*')):
    if path.is_file() and '__pycache__' not in path.parts:
        raw = path.read_bytes()
        files.append(dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
result = dict(schema='merge162-offline-verification-v1', actual_exit=0,
              original_merge_seal_unchanged=True, original_input_binding_checks=len(checked),
              original_162_exact=True, group_isolated_90_exact=True, overlap_examples=72,
              overlap_groups=48, tokenizer_checked=False, GPU_calls=0, training_started=False,
              scope='Local preparation only; not GPU trainer, tokenizer length proof or new semantic audit',
              files=files)
with (root / 'reviews/final-local-verification-01.json').open('x', encoding='utf-8') as out:
    json.dump(result, out, ensure_ascii=False, indent=2)
print(json.dumps({k: v for k, v in result.items() if k != 'files'}))
