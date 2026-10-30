"""New source-scope native targets and real deterministic-context token fixtures."""
import json,sys,time
from common import ROOT,PARENT
sys.path.append(str(PARENT))
from eval_runtime import prepare,preflight as native_preflight,TargetReached
from token_audit import tokenizer
from policy import gate_slots
import interface
def context_fixture(case,study,rollout):
    captured=[]
    def generate(messages):
        system=messages[0]['content'];body=json.loads(messages[-1]['content'])
        if not system.startswith('Execute only the selected Skill '):
            return json.dumps(dict(program=[dict(tool=t,arguments={}) for t in ('GET_FIELD_METADATA','INTERPRET_FIELD_NAME','GET_TABLE_FIELDS','SELECT_CATEGORY_TOPK','INFER_CATEGORY_FROM_TOPK')]))
        if 'selected Skill INTERPRET_FIELD_NAME.' in system:
            context=body['grounding_context'];value=dict(schema='grounded-source-v1',status='observed_description',field_name=context['field_name'],description=context['field_description'],source_consistency='not_adjudicated')
            return json.dumps(dict(summary='observed_description',data=dict(field_semantics=value)))
        if 'selected Skill SELECT_CATEGORY_TOPK.' in system:return '{"option_ids":["C001","C002","C003","C004","C005"],"summary":"CPU fixture only, not a prediction"}'
        if 'selected Skill INFER_CATEGORY_FROM_TOPK.' in system:
            captured.append(messages);raise TargetReached()
        raise AssertionError('unexpected CPU fixture stage')
    try:rollout(case['item'],study,generate,case['snapshot'])
    except TargetReached:pass
    assert len(captured)==1
    return interface.RecoverySession().rewrite(captured[0])
def run():
    from backend import EvaluationBackend
    original=json.loads((PARENT/'admission.json').read_bytes())
    evaluation=dict(original,evaluation_only=True,gpu_started_unix=time.time(),gpu_deadline_unix=time.time()+8*3600)
    backend=EvaluationBackend(evaluation);assert backend.runtime is None and len(backend.expected_tensors)==80
    native=native_preflight();study,cases,rollout=prepare();by={c['source_id']:c for c in cases}
    slots=json.loads((PARENT/'selection.json').read_bytes())['slots'];tok=tokenizer();lengths=[]
    def count(messages):
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        n=len(tok(text,add_special_tokens=False)['input_ids']);assert n+512<=49152;return n
    for slot in gate_slots(slots):lengths.append(dict(source_id=slot['source_id'],tokens=count(interface.RecoverySession().rewrite(by[slot['source_id']]['messages']))))
    context=[]
    for slot in slots:
        if slot['arm']!='N' or slot['source_index'] not in (4,10,12,13,17):continue
        messages=context_fixture(by[slot['source_id']],study,rollout);body=json.loads(messages[-1]['content'])
        assert 'decision_source_brief' not in body
        facts=body['evidence_boundary']['facts']
        eligible=sum(f.get('decisive_eligible') is True for f in facts.values())
        assert eligible>2
        assert not any(r['source_table_id'] in json.dumps(messages,ensure_ascii=False) for r in json.loads((ROOT/'source-scopes.json').read_bytes())['rows'])
        context.append(dict(slot=slot['slot_id'],tokens=count(messages),facts=eligible,cpu_fixture_not_prediction=True))
    assert len(context)==5
    return dict(native=native,initial_tokens=lengths,actual_scoped_context_tokens=context,new_model_calls=0,no_truncation=True,expected_fp32_tensors=80)
