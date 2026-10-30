"""Fixed denominators and native gate; no labels or model output rewriting."""
def unique(rows):
    by={r['slot_id']:r for r in rows}
    if len(by)!=len(rows):raise ValueError('duplicate slot IDs')
    return by
def gate_slots(slots):
    unique(slots)
    chosen=sorted([s for s in slots if s['arm']=='N' and 1<=s['source_index']<=20],key=lambda s:s['source_index'])
    if len(chosen)!=20 or len({s['source_id'] for s in chosen})!=20:raise ValueError('20 fixed unique gate sources required')
    return chosen
def gate_pass(rows,slots):
    by=unique(rows);expected=unique(slots)
    if set(by)!=set(expected):return False
    if any(not r.get('native_replay_exact') or not r.get('actual_target_exact') for r in rows):return False
    return sum(bool(r.get('legal_completion')) for r in rows)>=15
def pending_slots(slots,completed):
    by=unique(completed);unique(slots)
    if not set(by)<=set(unique(slots)):raise ValueError('unexpected completed slot')
    return [s for s in slots if s['slot_id'] not in by]
