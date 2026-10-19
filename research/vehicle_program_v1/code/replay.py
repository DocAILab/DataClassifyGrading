"""Replay completed traces against saved messages and responses without a model."""
import argparse
from copy import deepcopy
from pathlib import Path
from artifacts import BUDGET, load_controller, load_study, read
from runner import read_receipts


def replay(package, output):
    study, manifest = load_study(package)
    identity = read(Path(output) / 'run.json')
    if identity['study_sha256'] != manifest['study_sha256']:
        raise ValueError('replay study identity differs')
    receipts = read_receipts(output, study['expected_ids'])
    builder = load_controller()
    count = 0
    for n, row in enumerate(study['rows'], 1):
        if receipts.get(row['id'], {}).get('status') != 'completed':
            continue
        folder = Path(output) / 'sources' / f'{n:04d}'
        records = [read(p) for p in sorted((folder / 'calls').glob('*.json'))]
        position = 0
        def generate(messages):
            nonlocal position
            if position >= len(records) or messages != records[position]['messages']:
                raise AssertionError('replay message/order differs')
            call = records[position]
            position += 1
            if call['error_type'] is not None:
                raise AssertionError('completed source contains backend failure')
            generate.last_generation_metadata = deepcopy(call['generation'])
            return call['raw_response']
        controller = builder(generate, study['registry'], study['rows'], BUDGET, skill_pool='F66')
        trace = controller.run({'source_id': row['id'], 'domain': row['domain'],
                                'field_name': row['metadata']['field_name']}).to_dict()
        trace['execution_mode'] = identity['backend']
        if position != len(records) or trace != read(folder / 'trace.json'):
            raise ValueError('full controller replay differs')
        count += 1
    return {'completed_sources_replayed': count, 'receipt_count': len(receipts),
            'expected_count': len(study['rows']), 'model_calls': 0, 'pass': True}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    a = p.parse_args()
    import json
    print(json.dumps(replay(a.package, a.run)))
