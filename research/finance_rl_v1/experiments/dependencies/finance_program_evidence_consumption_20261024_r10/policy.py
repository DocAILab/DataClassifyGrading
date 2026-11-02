"""Fixed source identities, no outcome-dependent replacement or easy-case selection."""
TARGETS=(3,12,14,15,16,18,30,37)
def targeted(slots):return [s for s in slots if s['arm']=='N' and s['source_index'] in TARGETS]
def pending(slots,rows):
    done={r['slot_id'] for r in rows}
    return [s for s in slots if s['slot_id'] not in done]
