"""Independent recorded-I/O replay; never loads a GPU or a model."""
import argparse
from collections import Counter
from pathlib import Path
from io_utils import ROOT, read, put, sha, verify
from gate import enabled_for


def validate_record(q, x, route, arm, system, tokenizer, anchor=None):
    assert q['messages'] == x['messages']
    assert q['generation'] == x['generation']
    g=q['generation']
    assert g == {'do_sample':False,'max_new_tokens':512,'max_time':90.,'eos_token_id':[248044,248046],'pad_token_id':248044}
    enabled=enabled_for(arm,q['messages'],system)
    assert route['arm']==arm and route['enabled'] is enabled and route['layer_count']==40
    assert route['entered_disabled'] is (not enabled) and route['restored_disabled'] is True
    if anchor is not None:assert route['adapter_sha256']==anchor['adapter_sha256']
    assert x['parameter_devices']==['cuda:0'] and x['parameter_dtypes']==['torch.float16']
    if not x.get('error'):
        assert x['input_devices']==['cuda:0'] and 0<x['input_tokens']<=32768-512
        assert x['output_tokens']==len(x['generated_ids'])<=512
        assert x['truncation_at_budget'] is (x['output_tokens']>=512)
        if tokenizer is not None:
            prompt=tokenizer.apply_chat_template(q['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tokenizer(prompt,add_special_tokens=False)['input_ids']
            assert len(ids)==x['input_tokens']
            assert tokenizer.decode(x['generated_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)==x['raw_text']


class Recorded:
    def __init__(self, records):
        self.records=records;self.position=0;self.faults=[];self.last_generation_metadata={}
    def __call__(self,messages):
        self.last_generation_metadata={}
        try:
            assert self.position<len(self.records),'missing recorded call'
            r=self.records[self.position];self.position+=1
            assert r['request']['messages']==messages==r['result']['messages'],'recorded request differs'
        except AssertionError as e:
            self.faults.append(str(e));raise
        x=r['result']
        self.last_generation_metadata={k:x[k] for k in ('input_tokens','output_tokens','truncation_at_budget','error') if k in x}
        if x.get('error'):
            kind,sep,message=x['error'].partition(': ')
            assert sep and kind.isidentifier()
            raise type(kind,(Exception,),{})(message)
        return x['raw_text']
    def finish(self):
        assert not self.faults and self.position==len(self.records),'fault or unused model calls'


def audit_run(run, *, tokenizer=None):
    from program_loop import build_program_controller
    study=read(ROOT/'data/study.json');system=read(ROOT/'data/protocol.json')['planner_system']
    summary=read(Path(run)/'summary.json');ids=summary['expected_ids'];anchor=read(Path(run).parents[1]/'runs/training-anchor.json')
    assert ids==study['cohorts'][summary['cohort']] and len(ids)==len(set(ids))
    assert [s['id'] for s in summary['samples']]==ids
    counts=Counter();traces=0
    for s in summary['samples']:
        for arm in ('base','sft'):
            folder=Path(run)/'sources'/s['id']/arm
            receipt=read(folder/'receipt.json')
            assert receipt['id']==s['id'] and receipt['arm']==arm and receipt==s['arms'][arm]
            paths=sorted((folder/'calls').glob('call-*'))
            assert [p.name for p in paths]==['call-%04d'%i for i in range(1,len(paths)+1)]
            records=[]
            for p in paths:
                q,x,route=read(p/'request.json'),read(p/'result.json'),read(p/'route.json')
                assert route['request_sha256']==sha(p/'request.json') and route['result_sha256']==sha(p/'result.json')
                validate_record(q,x,route,arm,system,tokenizer,anchor)
                records.append({'request':q,'result':x});counts[arm+'_calls']+=1
                counts[arm+'_planner_enabled']+=int(route['enabled'])
                for key in ('input_tokens','output_tokens'):counts[arm+'_'+key]+=x.get(key) or 0
            replay=Recorded(records)
            row=next(r for r in study['rows'] if r['id']==s['id'])
            c=build_program_controller(replay,study['registry'],study['rows'],study['budget'],skill_pool='F66')
            assert len(c.library)==66
            actual=c.run({'source_id':row['id'],'domain':row['domain'],'field_name':row['metadata']['field_name']}).to_dict()
            replay.finish()
            assert actual==read(folder/'trace.json'),'full actual controller trace differs: '+s['id']+'/'+arm
            assert receipt['terminated']==actual['terminated'] and receipt['calls']==len(records)
            assert receipt['backend_errors']==sum(bool(r['result'].get('error')) for r in records)
            traces+=1
    assert summary['paired_sources']==len(ids)
    return {'full_trace_exact':True,'all_calls_consumed':True,'sources':len(ids),'traces':traces,
        'actual_route_and_tokens_verified':tokenizer is not None,**counts}


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--tokenizer',action='store_true');a=p.parse_args()
    m=verify();tok=None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(m['config']['model'],local_files_only=True)
    result=audit_run(a.run,tokenizer=tok);put(a.output,result);print(__import__('json').dumps(result))


if __name__=='__main__':main()
