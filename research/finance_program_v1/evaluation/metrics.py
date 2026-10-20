"""Post-run local-only labels; all failures retain the fixed denominator."""
# 评分始终遍历固定评估 ID，缺失或未合法终止的轨迹也计入分母；配对比较要求两组来源完全一致。
from collections import Counter


def macro(golds,preds,labels):
    values=[]
    for label in labels:
        tp=sum(g==label and p==label for g,p in zip(golds,preds))
        fp=sum(g!=label and p==label for g,p in zip(golds,preds))
        fn=sum(g==label and p!=label for g,p in zip(golds,preds))
        values.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return sum(values)/len(values) if values else 0.


def score(ids,traces,gold,registry):
    if len(ids)!=len(set(ids)) or not set(traces)<=set(ids) or not set(ids)<=set(gold):raise ValueError('score ID mismatch')
    counts=Counter();details={};golds=[];preds=[]
    # 遍历固定名单而非现存轨迹，缺失样本不会被静默排除并抬高准确率。
    for i in ids:
        t=traces.get(i) or {};final=t.get('final_result') or {};legal=t.get('terminated') is True
        prediction=final.get('category') if legal else None;grade=final.get('grading_level') if legal else None
        cat=bool(legal and prediction==gold[i]['category']);level=bool(legal and grade==gold[i]['grading_level'])
        history=[]
        for program in t.get('programs',[]):
            for step in program.get('steps',[]):
                data=step.get('data') or {}
                candidates=data.get('standard_candidates')
                if (step.get('call',{}).get('tool') in ('SELECT_CATEGORY_TOPK','SEARCH_STANDARD','SEARCH_STANDARD_BY_FIELD',
                        'SEARCH_STANDARD_BY_CONTEXT','SEARCH_STANDARD_BY_TABLE','SEARCH_STANDARD_BY_DATABASE','RESEARCH_STANDARD')
                        and step.get('status')=='success' and isinstance(candidates,list) and len(candidates)==5):
                    value=[v.get('category') or v.get('category_id') for v in candidates]
                    if len(set(value))==5:history.append(value)
        row={'terminated':legal,'category_correct':cat,'grading_correct':level,'joint_correct':cat and level,
            'top5_first_hit':bool(history and gold[i]['category'] in history[0]),
            'top5_final_hit':bool(history and gold[i]['category'] in history[-1]),
            'top5_any_hit':any(gold[i]['category'] in h for h in history)}
        details[i]=row;counts.update({k:int(v) for k,v in row.items()});golds.append(gold[i]['category']);preds.append(prediction)
    return {'denominator':len(ids),'missing':len(ids)-len(traces),**counts,
        'category_accuracy':counts['category_correct']/len(ids),
        'grading_accuracy':counts['grading_correct']/len(ids),'joint_accuracy':counts['joint_correct']/len(ids),
        'termination_rate':counts['terminated']/len(ids),
        'macro_f1_supported':macro(golds,preds,sorted(set(golds))),
        'macro_f1_fixed12':macro(golds,preds,sorted(registry)),'per_id':details}


def paired(ids,sft,base):
    if set(sft)!=set(base) or set(sft)!=set(ids):raise ValueError('paired IDs differ')
    result={}
    for key in next(iter(sft.values())):
        new=[i for i in ids if sft[i][key] and not base[i][key]]
        old=[i for i in ids if base[i][key] and not sft[i][key]]
        result[key]={'sft_only':new,'base_only':old,'delta':len(new)-len(old)}
    return result
