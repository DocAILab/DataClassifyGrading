"""Compare last recorded candidate coverage on the 44 outcome changes."""
from collections import Counter, defaultdict
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
NEW = BASE/'finance_program_grpo_val422_20261028_r4/reviews'
OLD = BASE/'finance_program_termination_val422_20261025_r2/reviews'
GOLD = BASE/'finance_program_student_val422_eval_20261017_r1/reviews/local-gold422.json'

def read(p):
    return json.loads(p.read_bytes())

def main():
    rows=read(NEW/'full422-score.json')['rows']; old={v['source_id']:v for v in read(OLD/'full422-score.json')['rows']}; refs=read(GOLD)
    groups=defaultdict(Counter); coverage=Counter(); errors={'SFT':Counter(),'GRPO':Counter()}
    for row in rows:
        before=old[row['source_id']]; reference=refs[row['source_id']]['category']
        tag='improved' if row['joint_correct'] and not before['joint_correct'] else 'worsened' if before['joint_correct'] and not row['joint_correct'] else 'other'
        presence=[]
        for name, item, root in (('SFT',before,OLD),('GRPO',row,NEW)):
            trace=read(root/'final-01/extract/runs'/item['slot_id']/'trace.json')
            candidates=trace['state'].get('standard_candidates') or []
            present=any(isinstance(c,dict) and c.get('category')==reference for c in candidates)
            presence.append(present);coverage[name]+=present
            for program in trace['programs']:
                for step in program['steps']:
                    if step['status']!='success':errors[name][(step['call']['tool'],step['status'])]+=1
        if tag!='other':groups[tag][str(tuple(presence))]+=1
    assert sum(groups['improved'].values())==20 and sum(groups['worsened'].values())==24
    result={'last_candidate_coverage':dict(coverage),'changed_last_candidate_transitions':{k:dict(v) for k,v in groups.items()},'executed_step_errors':{k:[{'tool':tool,'status':status,'count':n} for (tool,status),n in v.most_common()] for k,v in errors.items()}}
    with (ROOT/'candidate-compare-02.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
