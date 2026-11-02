"""Read-only aggregation of verified GRPO and historical SFT results."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib, json, statistics

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent
NEW = BASE/'finance_program_grpo_val422_20261028_r4/reviews'
OLD = BASE/'finance_program_termination_val422_20261025_r2/reviews'
TRAIN = BASE/'finance_program_grpo_full_migrated_20261027_r3/reviews/final-01/extract'
REF = BASE/'finance_program_student_val422_eval_20261017_r1/reviews/local-gold422.json'

def read(p): return json.loads(p.read_bytes())
def short(s): return s.split('/')[-1] if s else '无预测'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    a=read(NEW/'full422-score.json'); b=read(OLD/'full422-score.json'); refs=read(REF)
    old={v['source_id']:v for v in b['rows']}; rows=a['rows']
    assert len(rows)==len(old)==len(refs)==422 and {r['source_id'] for r in rows}==set(old)==set(refs)
    diffs=defaultdict(Counter);confusion=Counter(); presence=Counter(); bychange=defaultdict(Counter); trans=Counter(); cat_recall=defaultdict(Counter); changed=[]
    for r in rows:
        sid=r['source_id']; o=old[sid]; label=refs[sid]['category']; cat=short(label)
        y=bool(r['joint_correct']);z=bool(o['joint_correct']); p=r['prediction'] or {};op=o['prediction'] or {}
        diffs[cat]['total']+=1;diffs[cat]['old_correct']+=z;diffs[cat]['new_correct']+=y;diffs[cat]['improved']+=y and not z;diffs[cat]['worsened']+=z and not y
        if not y:confusion[(cat,short(p.get('category')))]+=1
        tag='improved' if y and not z else 'worsened' if z and not y else 'both_correct' if y else 'both_wrong'
        trans[(o['stop_reason'],r['stop_reason'])]+=1
        trace=read(NEW/'final-01/extract/runs'/r['slot_id']/'trace.json')
        cand=trace['state'].get('standard_candidates') or []
        contains=any(isinstance(v,dict) and v.get('category')==label for v in cand)
        if not y:
            presence['wrong_total']+=1;presence['no_prediction']+=not bool(p)
            if p:presence['wrong_pred_reference_in_final_candidates']+=contains;presence['wrong_pred_reference_not_in_final_candidates']+=not contains
        cat_recall[cat]['support']+=1;cat_recall[cat]['reference_in_final_candidates']+=contains
        bychange[tag][r['stop_reason']]+=1
        if tag in ('improved','worsened'):
            changed.append({'source_id':sid,'new_slot':r['slot_id'],'transition':tag,'reference_category':label,'sft_prediction':op,'grpo_prediction':p,'sft_outcome':o['stop_reason'],'grpo_outcome':r['stop_reason'],'reference_in_final_candidates':contains})
    c=read(TRAIN/'runs/completion.json'); batches=[];sources=defaultdict(list)
    for r in c['rows']:sources[r['source_id']].append(r)
    assert len(sources)==48 and all(len(r)==4 for r in sources.values()) and len(c['rows'])==192
    source_outcomes=Counter(sum(x['reward']['outcome'] for x in s) for s in sources.values())
    for v in c['batches']:
        rr=v['rows'];groups=defaultdict(list)
        for x in rr:groups[x['source_id']].append(x['reward']['A1'])
        batches.append({'batch':v['batch'],'legal':sum(x['legal_completion'] for x in rr),'correct':sum(x['reward']['outcome'] for x in rr),'mixed_sources':sum(len(set(x))>1 for x in groups.values()),'sources':len(groups)})
    ups=[]
    for u in c['updates']:
        m=u['micro'];ups.append({'version':u['policy_version'],'turns':u['planner_turns'],'tokens':u['action_tokens'],'grad_norm':u['grad_norm'],'update_norm':u['update_norm'],'minimum_ess':min(i['importance_relative_ESS'] for i in m),'maximum_cap_fraction':max(i['importance_cap_fraction'] for i in m),'old_current_max':max(i['max_abs_logp_delta'] for i in m)})
    result={'input_shas':{'new':sha(NEW/'full422-score.json'),'old':sha(OLD/'full422-score.json'),'training_completion':sha(TRAIN/'runs/completion.json'),'gold':sha(REF)},'cohort':422,'paired_by_reference_category':dict(diffs),'largest_error_confusions':[{'reference':x,'predicted':y,'count':n} for (x,y),n in confusion.most_common(18)],'candidate_last_state_audit':dict(presence),'candidate_coverage_by_reference':dict(cat_recall),'changed_outcomes':dict(bychange),'outcome_transitions':[{'sft':x,'grpo':y,'count':n} for (x,y),n in trans.most_common()],'training':{'source_outcome_success_counts':dict(source_outcomes),'batches':batches,'updates':ups},'changed_rows':changed}
    assert len(changed)==44 and presence['wrong_total']==288
    with (ROOT/'analysis-01.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='changed_rows'},ensure_ascii=False))
if __name__=='__main__':main()
