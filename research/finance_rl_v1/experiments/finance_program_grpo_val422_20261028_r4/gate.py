"""Fixed first20: strictly greater than70% means at least15 real predictions."""


def assess(rows):
    if len(rows)!=20 or len({r['slot_id'] for r in rows})!=20 or {r['source_index'] for r in rows}!=set(range(1,21)):
        raise ValueError('exact first20 unique sources required')
    valid=sum(bool(r.get('legal_completion') and isinstance(r.get('prediction'),dict)
                   and r['prediction'].get('category') and r['prediction'].get('grading_level')
                   and r.get('native_guard_valid')) for r in rows)
    exact=all(r.get('native_replay_exact') for r in rows)
    return dict(denominator=20,guard_valid_predictions=valid,legal_termination_rate=valid/20,
                required_predictions=15,criterion='strictly_greater_than_70_percent',
                algorithm_completed=sum(bool(r.get('algorithm_completed')) for r in rows),
                all_exact=exact,passed=valid>=15 and exact)
