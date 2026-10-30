import importlib.util,json,sys
from common import ROOT,PARENT,CAMPAIGN
import interface
spec=importlib.util.spec_from_file_location('same_native_preflight',CAMPAIGN/'preflight.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
def run():
    result=previous.run()
    from token_audit import tokenizer
    tok=tokenizer();lengths=[]
    for slot,stage in [('004-N','SELECT_CATEGORY_TOPK'),('009-N','INFER_CATEGORY_FROM_TOPK')]:
        for p in sorted((CAMPAIGN/'runs'/slot).glob('call-*/model-returned.json')):
            returned=json.loads(p.read_bytes())
            if returned['stage']!=stage:continue
            messages=json.loads((p.parent/'returned.json').read_bytes())['messages']
            actual=interface.rewrite(messages)
            text=tok.apply_chat_template(actual,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            n=len(tok(text,add_special_tokens=False)['input_ids']);assert n+512<=49152
            lengths.append(dict(slot=slot,stage=stage,tokens=n));break
    assert len(lengths)==2
    result['real_scope_fixture_tokens']=lengths;return result
if __name__=='__main__':print(json.dumps(run()))
