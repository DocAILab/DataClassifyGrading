import importlib.util,json
from common import ROOT,PREVIOUS
import interface
spec=importlib.util.spec_from_file_location('same_cpu_preflight',PREVIOUS/'preflight.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
def run():
    result=previous.run()
    from token_audit import tokenizer
    tok=tokenizer();lengths=[]
    for slot in ('014-N','018-N'):
        for p in sorted((PREVIOUS/'runs'/slot).glob('call-*/model-returned.json')):
            row=json.loads(p.read_bytes())
            if row['stage']!='INFER_CATEGORY_FROM_TOPK':continue
            m=json.loads((p.parent/'returned.json').read_bytes())['messages']
            actual=interface.rewrite(m);body=json.loads(actual[-1]['content'])
            if len(body['decision_source_brief'])<=2:continue
            text=tok.apply_chat_template(actual,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            n=len(tok(text,add_special_tokens=False)['input_ids']);assert n+512<=49152
            lengths.append(dict(slot=slot,tokens=n,eligible_fact_count=len(body['decision_source_brief'])));break
    assert len(lengths)==2;result['actual_context_fixture_tokens']=lengths;return result
if __name__=='__main__':print(json.dumps(run()))
