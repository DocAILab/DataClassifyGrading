"""Local-only Val422 scoring after every new generation and native verification."""
from collections import Counter
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from support import VAL,put,sha


def metrics(rows):
    n=len(rows)
    return dict(denominator=n,algorithm_completed=sum(r['algorithm_completed'] for r in rows),
                predictions=sum(r['legal_completion'] for r in rows),
                strict_native_success=sum(r['stop_reason']=='validated_terminal' for r in rows),
                best_effort_predictions=sum(r['stop_reason']=='best_effort_terminal' for r in rows),
                abstentions=sum(r['stop_reason']=='abstained_terminal' for r in rows),
                category_correct=sum(r['category_correct'] for r in rows),grade_correct=sum(r['grade_correct'] for r in rows),
                joint_correct=sum(r['joint_correct'] for r in rows),stop_reasons=dict(Counter(r['stop_reason'] for r in rows)),
                model_calls=sum(r['model_calls'] for r in rows))


def main():
    proof=json.loads((ROOT/'reviews/independent-verification.json').read_bytes())
    assert proof['passed'] and proof['new_native_replays']==422
    ex=ROOT/'reviews/final-01/extract';source=ex/'runs/completion.json'
    c=json.loads(source.read_bytes());rows=c['rows']
    assert len(rows)==422 and len({r['source_id'] for r in rows})==422
    refpath=ROOT.parent/'finance_program_student_val422_eval_20261017_r1/reviews/local-gold422.json';refs=json.loads(refpath.read_bytes())
    assert {r['source_id'] for r in rows}==set(refs)
    registry=json.loads((ROOT/'eval-inputs.json').read_bytes())['study']['registry']
    counts={k:dict(tp=0,fp=0,fn=0,support=0) for k in registry}
    for r in rows:
        ref=refs[r['source_id']];pred=r['prediction'] or {}
        r['category_correct']=pred.get('category')==ref['category'];r['grade_correct']=pred.get('grading_level')==ref['grading_level']
        r['joint_correct']=r['category_correct'] and r['grade_correct']
        counts[ref['category']]['support']+=1
        if r['category_correct']:counts[ref['category']]['tp']+=1
        else:
            counts[ref['category']]['fn']+=1
            if pred.get('category') in counts:counts[pred['category']]['fp']+=1
    for v in counts.values():
        denom=2*v['tp']+v['fp']+v['fn'];v['f1']=2*v['tp']/denom if denom else 0
    supported=[v for v in counts.values() if v['support']]
    old=ROOT.parent/'finance_program_termination_val422_20261025_r2/reviews/full422-score.json'
    baseline_proof=json.loads((ROOT/'reviews/sft-baseline-identity-01.json').read_bytes())
    historical_manifest=json.loads((old.parent/'delivery-manifest.json').read_bytes())
    score_entry=next(x for x in historical_manifest['files'] if x['path']=='reviews/full422-score.json')
    assert baseline_proof['passed'] and baseline_proof['historical_result_rows']==422 and sha(old)==score_entry['sha256']
    baseline=json.loads(old.read_bytes());oldrows={x['source_id']:x for x in baseline['rows']}
    assert len(oldrows)==422 and set(oldrows)=={r['source_id'] for r in rows}
    comparisons=dict(improved=sum(r['joint_correct'] and not oldrows[r['source_id']]['joint_correct'] for r in rows),
                     worsened=sum(not r['joint_correct'] and oldrows[r['source_id']]['joint_correct'] for r in rows),
                     both_correct=sum(r['joint_correct'] and oldrows[r['source_id']]['joint_correct'] for r in rows),
                     both_wrong=sum(not r['joint_correct'] and not oldrows[r['source_id']]['joint_correct'] for r in rows))
    result=dict(kind='GRPO_only_fullVal422_generation',all=metrics(rows),paired_comparison=comparisons,baseline_score_sha256=sha(old),
                strata={k:metrics([r for r in rows if r['stratum']==k]) for k in sorted({r['stratum'] for r in rows})},
                by_outcome={k:metrics([r for r in rows if r['stop_reason']==k]) for k in sorted({r['stop_reason'] for r in rows})},
                macro_f1_12=sum(v['f1'] for v in counts.values())/len(counts),
                macro_f1_supported=sum(v['f1'] for v in supported)/len(supported),supported_class_count=len(supported),per_category=counts,
                rows=rows,reference_sha256=sha(refpath),completion_sha256=sha(source),
                new_native_replays=422,new_actual_model_calls=proof['new_actual_model_calls'],N_only=True,reused_results=0,
                independent_fresh_test=False,semantic_quality_verified_all=False)
    put(ROOT/'reviews/full422-score.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','per_category')},ensure_ascii=True))


if __name__=='__main__':main()
