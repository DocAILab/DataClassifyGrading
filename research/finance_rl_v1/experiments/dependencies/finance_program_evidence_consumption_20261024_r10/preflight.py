"""Zero-model CPU native and real request tokenizer acceptance before admission."""
import json,sys
from common import ROOT,PARENT,BASELINE
sys.path.append(str(PARENT))
from eval_runtime import prepare,preflight as native_preflight
from token_audit import tokenizer
import interface
from policy import targeted
def run():
    native=native_preflight();study,cases,rollout=prepare();by={c['source_id']:c for c in cases}
    from program_loop.validation import parse_program
    from agent.orchestration.contracts import MetadataRecord
    import runtime
    engine=__import__('eval_runtime').parent.load_engine(__import__('eval_runtime').parent.E2)
    # Verify the actual new controller and native Program/Planner limits without generation.
    controller=runtime.previous.build(engine).__globals__['old'].c.controller(lambda m:'{}',study)
    assert len(parse_program(json.dumps(dict(program=[dict(tool='GET_FIELD_METADATA',arguments={}) for _ in range(13)])),controller.bridge.catalog,{},16).steps)==13
    trace=rollout(cases[0]['item'],study,lambda m:'{}',cases[0]['snapshot'])
    assert len(trace['planner_attempts'])==10
    tok=tokenizer();initial=[];changed=[]
    def count(messages):
        value=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        n=len(tok(value,add_special_tokens=False)['input_ids']);assert n+512<=49152
        return n
    slots=json.loads((PARENT/'selection.json').read_bytes())['slots']
    for slot in targeted(slots):
        initial.append(dict(slot=slot['slot_id'],tokens=count(interface.RecoverySession().rewrite(by[slot['source_id']]['messages']))))
    fixture=BASELINE/'reviews/final-01/extract/runs'
    if not fixture.exists():fixture=BASELINE/'runs'
    groups=json.loads((BASELINE/'source-scopes.json').read_bytes())['rows']
    for slot in targeted(slots):
        session=interface.RecoverySession()
        for p in sorted((fixture/slot['slot_id']).glob('call-*/returned.json')):
            source=json.loads(p.read_bytes())['messages'];messages=[dict(role=m['role'],content=m['content']) for m in source]
            actual=session.rewrite(messages);body=json.loads(actual[-1]['content'])
            if 'acquired_evidence_delta' not in body:continue
            assert body['evidence_boundary']['facts']==json.loads(messages[-1]['content'])['evidence_boundary']['facts']
            assert not any(g['source_table_id'] in json.dumps(actual,ensure_ascii=False) for g in groups)
            changed.append(dict(slot=slot['slot_id'],call=p.parent.name,tokens=count(actual),added_facts=len(body['acquired_evidence_delta']['added_fact_ids'])))
    assert len(changed)==11
    from backend import EvaluationBackend
    import time
    admission=json.loads((PARENT/'admission.json').read_bytes());admission.update(evaluation_only=True,gpu_deadline_unix=time.time()+3600)
    model=EvaluationBackend(admission);assert model.runtime is None and len(model.expected_tensors)==80
    return dict(native=native,native13steps=True,native10planner=True,initial_tokens=initial,changed_classifier_tokens=changed,
                no_truncation=True,expected_fp32_tensors=80,new_model_calls=0,gpu_loaded=False)
