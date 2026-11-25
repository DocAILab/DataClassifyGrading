"""Export the fixed, label-free native evaluation inputs. Never export dossiers."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re

from training_contract import MANIFEST_SHA256, ProtocolError, file_sha256, sha256_bytes

FIELDS = {'database_name', 'database_description', 'table_name', 'table_description',
          'field_name', 'field_description', 'field_type', 'value'}


def raw_bytes(value):
    # Native H's serialization preserves mapping insertion order, including PromptRefs.
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def exclusive_json(path, value):
    """Durable native-order JSON; raw messages/PromptRefs keep their original order."""
    with Path(path).open('xb') as handle:
        handle.write(raw_bytes(value) + b'\n')
        handle.flush()
        os.fsync(handle.fileno())


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def read_locked(base, relative, expected):
    base = Path(base).resolve(strict=True)
    path = base / relative
    require(path.resolve(strict=True).is_relative_to(base) and not path.is_symlink() and path.is_file(),
            'input path escapes physical source root')
    require(file_sha256(path) == expected, 'locked input SHA mismatch: ' + relative)
    return json.loads(path.read_bytes())


def build_inputs(root, source):
    """Pure read/export construction; actual prepare is a separate exclusive step."""
    root, source = Path(root), Path(source)
    selection_path = root / 'data/selection-manifest.json'
    require(file_sha256(selection_path) == MANIFEST_SHA256, 'fixed selection manifest SHA mismatch')
    selection = json.loads(selection_path.read_bytes())
    expected = selection['input_sha256']
    original = read_locked(source, 'inputs.json', expected['inputs.json'])
    hashes = {'inputs.json': expected['inputs.json']}
    data = {}
    for name in ('eval-initial18.jsonl', 'recovery-eval.json'):
        path = root / 'data' / name
        require(not path.is_symlink() and file_sha256(path) == selection['outputs_sha256'][name],
                'fixed evaluation input SHA mismatch')
        hashes['data/' + name] = file_sha256(path)
        data[name] = ([json.loads(line) for line in path.read_bytes().splitlines()]
                      if name.endswith('jsonl') else json.loads(path.read_bytes())['rows'])
    study = {k: deepcopy(original['study'][k]) for k in ('registry', 'rows', 'budget')}
    require(len(study['rows']) == 200, 'all original 200 native metadata rows required')
    for row in study['rows']:
        require(set(row) == {'id', 'domain', 'metadata'} and set(row['metadata']) == FIELDS,
                'native study rows may only contain canonical metadata')
    require(len({row['id'] for row in study['rows']}) == 200, 'native row identities must be unique')
    for value in study['registry'].values():
        require(set(value) == {'name', 'path', 'description', 'grading_level'},
                'public taxonomy/frozen grade only')
    cases = []
    for split, name in (('initial', 'eval-initial18.jsonl'), ('recovery', 'recovery-eval.json')):
        for row in data[name]:
            source_id = row['source_id']
            original_item = original['items']['initial-' + source_id]
            require(original_item['kind'] == 'initial' and original_item['source_id'] == source_id,
                    'evaluation source must bind its actual initial item')
            item = {k: deepcopy(original_item[k]) for k in ('id', 'source_id', 'kind', 'initial')}
            snapshot = deepcopy(original['snapshots'][item['id']])
            require(snapshot['prefix_calls'] == 0 and not snapshot['prefix_steps'],
                    'only original from-zero snapshot allowed')
            prefix = []
            if split == 'recovery':
                context_path = Path(row['context_file'])
                require(context_path.name == 'context.json' and
                        re.fullmatch(r'call-\d+', context_path.parent.name) is not None,
                        'actual target call directory required')
                target_number = int(context_path.parent.name.split('-')[1])
                target = read_locked(source, row['context_file'], expected[row['context_file']])
                hashes[row['context_file']] = expected[row['context_file']]
                require(raw_bytes(target['student_messages']) == raw_bytes(row['messages']),
                        'fixed recovery messages differ from actual target context')
                directories = sorted((source / context_path.parent.parent).glob('call-*'),
                                     key=lambda p: int(p.name.split('-')[1]))
                prior = [p for p in directories if int(p.name.split('-')[1]) < target_number]
                require([int(p.name.split('-')[1]) for p in prior] == list(range(1, target_number)),
                        'missing or reordered native prefix call')
                for directory in prior:
                    context_file = (directory / 'context.json').relative_to(source).as_posix()
                    record_file = (directory / 'record.json').relative_to(source).as_posix()
                    context = read_locked(source, context_file, expected[context_file])
                    record = read_locked(source, record_file, expected[record_file])
                    generation = record['generation']
                    require(set(generation) <= {'input_tokens', 'output_tokens', 'error', 'truncation_at_budget'},
                            'unexpected private provider generation fields')
                    prefix.append(dict(messages=deepcopy(context['student_messages']),
                                       response=record['response'], generation=deepcopy(generation), error=record['error'],
                                       context_file=context_file, context_sha256=expected[context_file],
                                       record_file=record_file, record_sha256=expected[record_file]))
                    hashes.update({context_file: expected[context_file], record_file: expected[record_file]})
            cases.append(dict(source_id=source_id, kind=split, item=item, snapshot=snapshot,
                              messages=deepcopy(row['messages']), prefix=prefix))
    require([len(data[n]) for n in ('eval-initial18.jsonl', 'recovery-eval.json')] == [18, 12],
            'fixed 18 initial / 12 recovery denominator required')
    receipt = json.loads((root / 'reviews/runtime-copy-receipt-01.json').read_bytes())
    vendor = {r['target']: r['sha256'] for r in receipt['files']}
    require(len(vendor) == 121, 'exact original 121 vendor assets required')
    for path, digest in vendor.items():
        require(path.startswith('vendor/') and file_sha256(root / path) == digest,
                'native vendor bytes differ from copied lock')
    return dict(schema='b80-native-paired-inputs-v1', study=study, cases=cases), dict(
        schema='b80-runtime-input-manifest-v1', selection_manifest_sha256=MANIFEST_SHA256,
        original_files_sha256=hashes, vendor_files_sha256=vendor,
        source_order=[c['source_id'] for c in cases], counts=dict(initial=18, recovery=12),
        remote_allowlist=['runtime-inputs/inputs.json', 'runtime-inputs/manifest.json'])


def prepare_runtime(root, source):
    inputs, manifest = build_inputs(root, source)
    destination = Path(root) / 'runtime-inputs'
    require(not destination.exists() and not destination.is_symlink(), 'runtime-inputs already exists; no overwrite')
    destination.mkdir(exist_ok=False)
    payload = raw_bytes(inputs)
    with (destination / 'inputs.json').open('xb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    manifest['inputs_sha256'] = sha256_bytes(payload)
    exclusive_json(destination / 'manifest.json', manifest)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_runtime(args.root, args.source), ensure_ascii=False))
