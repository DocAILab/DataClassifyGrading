"""Keep every N100 denominator and immutable reuse; never replace failures."""
def pending(slots,reused):
 if len(slots)!=100 or any(s['arm']!='N' for s in slots):raise ValueError('exact100 N-only required')
 if len({s['slot_id'] for s in slots})!=100:raise ValueError('duplicate slot')
 old={r['slot_id']:r for r in reused}
 if len(old)!=8:raise ValueError('exact8 immutable previous results required')
 for s in slots:
  if s['slot_id'] in old:
   if any(old[s['slot_id']][k]!=s[k] for k in ('source_id','group','arm','source_index')):raise ValueError('reuse identity mismatch')
 return [s for s in slots if s['slot_id'] not in old]
