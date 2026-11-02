"""Pure two/three-card memory admission; no process or device mutation.

The envelopes are conservative engineering estimates, not measured B80 peaks.
Only an actual first forward/backward can establish fit. It is part of the
single formal run, not a discarded update-bearing trial or permission to retry.
"""
import itertools
import math
import time

GIB = 1024 ** 3
HEADROOM = 2 * GIB
PROFILES = {
    3: {'allocator': 18 * GIB, 'model_weights': 8 * GIB},
}


def _validated(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('nonempty actual GPU snapshot required')
    ids, uuids = set(), set()
    for row in rows:
        if (type(row['index']) is not int or row['index'] < 0 or row['index'] in ids
                or not isinstance(row['uuid'], str) or not row['uuid'].startswith('GPU-')
                or row['uuid'] in uuids):
            raise ValueError('ambiguous actual GPU identity')
        ids.add(row['index']); uuids.add(row['uuid'])
        if (type(row['free_bytes']) is not int or type(row['total_bytes']) is not int
                or not 0 <= row['free_bytes'] <= row['total_bytes']
                or not isinstance(row['other_compute_pids'], list)
                or any(type(p) is not int or p <= 0 for p in row['other_compute_pids'])):
            raise ValueError('invalid actual GPU memory/process snapshot')
    return sorted(rows, key=lambda row: row['index'])


def candidates(rows, *, max_sequence_tokens):
    """Return stable preference order; caller must acquire all locks and recheck."""
    if type(max_sequence_tokens) is not int or not 0 < max_sequence_tokens <= 32768:
        raise ValueError('actual untruncated token audit within 32768 required')
    rows = _validated(rows)
    result = []
    for count, profile in PROFILES.items():
        eligible = [r for r in rows if r.get('bf16_compatible') is True
                    and r['free_bytes'] >= profile['allocator'] + HEADROOM]
        for combo in itertools.combinations(eligible, count):
            occupied = sum(bool(r['other_compute_pids']) for r in combo)
            result.append({
                'gpu_indices': [r['index'] for r in combo],
                'gpu_uuids': [r['uuid'] for r in combo],
                'allocator_limit_bytes': [profile['allocator']] * count,
                'model_max_memory_bytes': [profile['model_weights']] * count,
                'headroom_bytes': HEADROOM,
                'max_sequence_tokens': max_sequence_tokens,
                'admission_kind': 'shared_provisional_envelope' if occupied else 'free_provisional_envelope',
                'occupied_card_count': occupied,
                'snapshot': [dict(r) for r in combo],
                'scientific_updates_allowed': 10,
            })
    return sorted(result, key=lambda r: (r['occupied_card_count'], len(r['gpu_indices']), r['gpu_indices']))


def recheck(rows, anchor):
    """Validate identity and headroom again after the caller holds every lock."""
    current = {r['index']: r for r in _validated(rows)}
    wanted = candidates(list(current.values()), max_sequence_tokens=anchor['max_sequence_tokens'])
    match = [c for c in wanted if c['gpu_indices'] == anchor['gpu_indices']
             and c['gpu_uuids'] == anchor['gpu_uuids']]
    if len(match) != 1 or any(match[0][k] != anchor[k] for k in (
            'allocator_limit_bytes', 'model_max_memory_bytes', 'headroom_bytes')):
        raise ValueError('locked GPU identity/memory admission no longer valid')
    return match[0]


def remaining_seconds(admission, *, now=None):
    deadline = admission['gpu_deadline_unix']
    now = time.time() if now is None else now
    if (type(deadline) not in (int, float) or type(now) not in (int, float)
            or not math.isfinite(deadline) or not math.isfinite(now)):
        raise ValueError('finite absolute GPU deadline required')
    remaining = deadline - now
    if remaining <= 0:
        raise TimeoutError('cumulative GPU deadline exhausted; never reset per stage')
    return remaining
