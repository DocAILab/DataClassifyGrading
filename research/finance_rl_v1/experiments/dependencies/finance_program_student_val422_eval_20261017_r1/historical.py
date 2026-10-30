"""No-API replay regression of a prior real grounded training-source trajectory."""
import argparse
from pathlib import Path
import json
from io_utils import ROOT,read,put,sha


def prepare():
    source=ROOT.parent/'finance_program_student_sft_20261016_r1/reviews/dataset-provenance-01.json'
    item=read(source)['rows'][0];path=Path(item['trace_path']);assert sha(path)==item['trace_sha256']
    folder=path.parent;study=read(ROOT.parent/'finance_program_teacher_balanced_20261016_r9/study.json')
    records=[]
    for p in sorted((folder/'calls').glob('call-*')):
        q,x=read(p/'request.json'),read(p/'result.json');assert x['complete']
        records.append({'request':{'messages':q['executor_messages']},
            'result':{'messages':q['executor_messages'],'raw_text':x['content']}})
    assert records
    put(ROOT/'fixtures/historical.json',{'study':study,'trace':read(path),'records':records,
        'source_id':item['source_id'],'original_trace_sha256':item['trace_sha256']})


def audit():
    from program_loop import build_program_controller
    from replay import Recorded
    fixture=read(ROOT/'fixtures/historical.json');s=fixture['study'];i=fixture['source_id']
    row=next(r for r in s['rows'] if r['id']==i);backend=Recorded(fixture['records'])
    c=build_program_controller(backend,s['registry'],s['rows'],s['budget'],skill_pool='F66')
    actual=c.run({'source_id':i,'domain':row['domain'],'field_name':row['metadata']['field_name']}).to_dict()
    backend.finish();assert actual==fixture['trace'],'historical real trace replay differs'
    result={'historical_real_grounded_full_replay':True,'calls':len(fixture['records']),
        'source_id':i,'original_trace_sha256':fixture['original_trace_sha256'],'gpu_loaded':False,'new_api_calls':0}
    put(ROOT/'reviews/historical-proof.json',result);print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','audit'));a=p.parse_args()
    prepare() if a.mode=='prepare' else audit()
