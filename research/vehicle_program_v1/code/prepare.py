"""Freeze the vehicle Val study and mechanically isolate evaluator-only labels."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from artifacts import ROOT, REVISION, BUDGET, GENERATION, code_inventory, put, read, require_private_output, sha, verify_vendor

SIZES = {'SFT': 1000, 'TRAIN': 1105, 'VAL': 397, 'TEST': 395}


def project(row):
    metadata = row['metadata']
    if not isinstance(row.get('id'), str) or not row['id'] or not isinstance(row.get('domain'), str) or not row['domain'].strip():
        raise ValueError('missing source identity/domain')
    if not isinstance(metadata.get('field_name'), str) or not metadata['field_name'].strip():
        raise ValueError('missing field name')
    result = {'field_name': metadata['field_name']}
    description = metadata.get('field_description')
    if isinstance(description, str) and description.strip():
        result['field_description'] = description
    return {'id': row['id'], 'domain': row['domain'], 'metadata': result}


def registry_from(corpus):
    registry = {}
    for entry in corpus:
        ident = entry['id']
        if not isinstance(ident, str) or not ident or ident in registry:
            raise ValueError('invalid/duplicate standard ID')
        required = ('category_root_level', 'category_leaf_level', 'sensitivity_level', 'data_description_and_example', 'reference_standard')
        if any(not isinstance(entry[k], str) or not entry[k].strip() for k in required):
            raise ValueError('standard missing required text')
        registry[ident] = {'standard_id': ident, 'path': [entry['category_root_level'], entry['category_leaf_level']],
                           'name': entry['category_leaf_level'], 'grading_level': entry['sensitivity_level'],
                           'description': entry['data_description_and_example'] + '\n' + entry['reference_standard']}
    if not registry:
        raise ValueError('empty standard corpus')
    return registry


def gold_for(rows, registry):
    answers = {}
    for row in rows:
        path = [row['classification']['category_root_level'], row['classification']['category_leaf_level']]
        grade = row['grading']['sensitivity_level']
        compatible = [key for key, e in registry.items() if e['path'] == path and e['grading_level'] == grade]
        if not compatible:
            raise ValueError('gold has no compatible standard')
        answers[row['id']] = {'path': path, 'grading_level': grade, 'compatible_standard_ids': compatible}
    return answers


def prepare(source, output):
    source, output = Path(source), Path(output)
    require_private_output(output)
    verify_vendor()
    pinned = read(ROOT / 'pinned-inputs.json')
    if pinned['revision'] != REVISION:
        raise ValueError('data revision differs')
    for name, expected in pinned['files'].items():
        path = source / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError('pinned input hash differs: ' + name)
    splits = {key: read(source / f'DCG_VEHICLE/DCG_VEHICLE_{key}.json') for key in SIZES}
    ids = {}
    for key, rows in splits.items():
        ids[key] = [r['id'] for r in rows]
        if len(rows) != SIZES[key] or len(set(ids[key])) != len(rows):
            raise ValueError('split size/unique IDs differ: ' + key)
    overlaps = {a + '/' + b: len(set(ids[a]) & set(ids[b])) for a, b in combinations(ids, 2)}
    if any(overlaps.values()):
        raise ValueError('source split overlap')
    corpus = read(source / 'DCG_VEHICLE/DCG_VEHICLE_CORPUS.json')
    registry = registry_from(corpus)
    rows = [project(r) for r in splits['VAL']]
    answers = gold_for(splits['VAL'], registry)
    study = {'schema': 'vehicle-nontd-v1', 'dataset_revision': REVISION, 'scope': 'development Val397, not blind Test',
             'skill_pool': 'F66', 'budget': BUDGET, 'generation': GENERATION,
             'expected_ids': ids['VAL'], 'rows': rows, 'registry': registry,
             'candidate_unit': 'standard_entry; multiple entries may share category and grade',
             'initial_visible_fields': ['domain', 'field_name'],
             'tool_accessible_fields': ['field_name', 'field_description']}
    output.mkdir(parents=True, exist_ok=False)
    put(output / 'inference/study.json', study)
    put(output / 'evaluation/gold.json', {'answers': answers})
    report = {'split_sizes': {k: len(v) for k, v in splits.items()}, 'id_overlaps': overlaps,
              'standard_entries': len(registry), 'leaf_paths': len({tuple(e['path']) for e in registry.values()}),
              'multistandard_leaf_paths': sum(n > 1 for n in Counter(tuple(e['path']) for e in registry.values()).values()),
              'gold_compatible_entry_counts': dict(Counter(len(v['compatible_standard_ids']) for v in answers.values())),
              'val_rows': len(rows), 'field_description_present': sum('field_description' in r['metadata'] for r in rows),
              'table_or_database_context': False, 'real_model_run': False, 'training_run': False}
    put(output / 'preparation-report.json', report)
    put(output / 'manifest.json', {'schema': 1, 'repository': 'DocAILab/DCG', 'revision': REVISION,
                                   'source_files': pinned['files'], 'code': code_inventory(),
                                   'study_sha256': sha(output / 'inference/study.json'),
                                   'gold_sha256': sha(output / 'evaluation/gold.json')})
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    import json
    print(json.dumps(prepare(args.source, args.output), ensure_ascii=False))
