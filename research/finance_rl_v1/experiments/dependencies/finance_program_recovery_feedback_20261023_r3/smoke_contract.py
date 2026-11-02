DEADLINE=1792770117.3153024
MAX_CALLS=168
def select(slots):
    chosen=[s for s in slots if s['arm']=='N' and 1<=s['source_index']<=4]
    if len(chosen)!=4 or len({s['source_index'] for s in chosen})!=4:raise ValueError('four unique fixed N sources required')
    return sorted(chosen,key=lambda s:s['source_index'])
def can_start(*,now,parent_live):return not parent_live and now<DEADLINE-600
