"""Fixed full422 contracts. No inference, labels fetched remotely, or model imports."""
import hashlib
from pathlib import Path, PurePosixPath
import shutil
from metrics import score, paired

ARMS = ('base', 'sft')
ANCHOR_KEYS = ('adapter_sha256', 'adapter_config_sha256', 'training_manifest_sha256',
               'completion_sha256', 'cpu_proof_sha256')


def no_labels(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.lower() in {'ground_truth', 'ground_truth_level', 'classification', 'grading',
                               'category', 'category_id', 'grading_level', 'label', 'labels', 'label_status'}:
                raise ValueError('label in model metadata: ' + key)
            no_labels(item)
    elif isinstance(value, list):
        for item in value: no_labels(item)


def partition(study, train_ids=()):
    rows = study['rows']; no_labels(rows)
    ids = [r['id'] for r in rows]; old = list(study['cohorts']['val80'])
    if (len(ids) != 422 or len(set(ids)) != 422 or len(old) != 80 or len(set(old)) != 80
            or not set(old) <= set(ids) or set(ids) & set(train_ids)):
        raise ValueError('expected exact unique422, fixed80 subset, and no training overlap')
    for ident in ids:
        if not isinstance(ident, str) or not ident or any(c in ident for c in '/\\:') or ident in ('.', '..'):
            raise ValueError('unsafe source ID')
    remaining = [i for i in ids if i not in set(old)]
    smoke = study['cohorts']['smoke']
    if len(smoke) != 2 or len(set(smoke)) != 2 or not set(smoke) <= set(remaining):
        raise ValueError('old engineering smoke must occur only in new formal342')
    result = {'val80': old, 'val342': remaining, 'full422': old + remaining}
    for key, values in result.items():
        if key in study['cohorts'] and study['cohorts'][key] != values:
            raise ValueError('frozen cohort order differs: ' + key)
    return result


def reuse_identity(old, new, old_files, new_files):
    for key in set(old) | set(new):
        if key != 'cohorts' and old.get(key) != new.get(key):
            raise ValueError('source study identity differs: ' + key)
    expected = dict(old['cohorts']); expected.update(partition(old))
    if new['cohorts'] != expected: raise ValueError('only supplementary cohorts may change')
    protected = lambda n: n.startswith(('runtime/', 'legacy/')) or n in (
        'backend.py', 'replay.py', 'gate.py', 'io_utils.py', 'metrics.py',
        'verify_training.py', 'historical.py', 'data/protocol.json', 'model-identity.json')
    names = {n for n in old_files if protected(n)} | {n for n in new_files if protected(n)}
    for name in names:
        if name not in old_files or name not in new_files or old_files[name] != new_files[name]:
            raise ValueError('immutable implementation identity differs: ' + name)


def anchor_identity(expected, actual):
    for key in ANCHOR_KEYS:
        if not expected.get(key) or expected[key] != actual.get(key):
            raise ValueError('training identity differs: ' + key)


def copy_exclusive(source, target):
    target = Path(target); target.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open('rb') as src, target.open('xb') as dst:
        shutil.copyfileobj(src, dst)


def safe_members(archive, expected, member_sha):
    entries = archive.getmembers(); names = [m.name for m in entries]
    if len(names) != len(set(names)) or set(names) != set(expected) | {'archive-members.json'}:
        raise ValueError('archive member list mismatch')
    for item in entries:
        p = PurePosixPath(item.name)
        if (not item.isfile() or p.is_absolute() or '..' in p.parts or '\\' in item.name
                or ':' in item.name or not p.parts or any(x in ('', '.', '..') for x in p.parts)
                or p.as_posix() != item.name):
            raise ValueError('unsafe archive member: ' + item.name)
        h = hashlib.sha256()
        with archive.extractfile(item) as stream:
            for block in iter(lambda: stream.read(1048576), b''): h.update(block)
        wanted = member_sha if item.name == 'archive-members.json' else expected[item.name]['sha256']
        if h.hexdigest() != wanted: raise ValueError('archive member SHA differs: ' + item.name)
        if item.name != 'archive-members.json' and item.size != expected[item.name]['bytes']:
            raise ValueError('archive member size differs: ' + item.name)
    return entries


def merge_traces(parts, old, supplementary, *, complete=True):
    result = {}
    if set(old) != set(ARMS) or set(supplementary) != set(ARMS): raise ValueError('arms differ')
    for arm in ARMS:
        for values, key in ((old[arm], 'val80'), (supplementary[arm], 'val342')):
            if not set(values) <= set(parts[key]) or (complete and set(values) != set(parts[key])):
                raise ValueError('formal source coverage differs: ' + key + '/' + arm)
        result[arm] = {**old[arm], **supplementary[arm]}
    return result


def combined_metrics(parts, traces, gold, registry, *, complete):
    ids = parts['full422']
    if len(ids) != 422 or len(set(ids)) != 422 or set(traces) != set(ARMS):
        raise ValueError('full422 contract differs')
    if set(gold) != set(ids): raise ValueError('full gold binding differs')
    for arm in ARMS:
        if not set(traces[arm]) <= set(ids) or (complete and set(traces[arm]) != set(ids)):
            raise ValueError('incomplete or unexpected full traces')
    metrics = {arm: score(ids, traces[arm], gold, registry) for arm in ARMS}
    result = {'evaluation_complete': bool(complete), 'formal_denominator_per_arm': 422,
              'expected_formal_traces': 844, 'observed_formal_traces': sum(map(len, traces.values())),
              'missing_per_arm': {arm: 422 - len(traces[arm]) for arm in ARMS}}
    if not complete:
        result['as_available_metrics'] = metrics
        return result
    result.update(formal_traces=844, metrics=metrics,
                  paired=paired(ids, metrics['sft']['per_id'], metrics['base']['per_id']),
                  cohorts={key: {arm: score(parts[key], {i:t for i,t in traces[arm].items() if i in set(parts[key])},
                                           gold, registry) for arm in ARMS} for key in ('val80', 'val342')})
    return result
