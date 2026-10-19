"""One pair per source, real frozen F66 execution; no labels or scoring."""
import argparse
import datetime
from pathlib import Path
import time
from io_utils import ROOT,read,put,verify


def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',choices=('smoke','val80'),required=True);a=p.parse_args()
    m=verify();c=m['config'];study=read(ROOT/'data/study.json')
    anchor=read(ROOT/'runs/training-anchor.json')
    if anchor['training_manifest_sha256']!=c['training_manifest_sha256']:raise ValueError('training anchor differs')
    output=ROOT/'runs'/a.cohort;output.mkdir(parents=True,exist_ok=False)
    ids=study['cohorts'][a.cohort];rows={r['id']:r for r in study['rows']};started=time.monotonic()
    summary={'cohort':a.cohort,'expected_ids':ids,'samples':[{'id':i,'arms':{}} for i in ids],
        'paired_sources':0,'status':'running','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    from backend import RoutedBackend
    from program_loop import build_program_controller
    # Separate processes per cohort, same frozen weights and routing; backend-init artifacts get cohort-specific path.
    try:
        backend=RoutedBackend(c['model'],c['adapter'],anchor,a.cohort)
        for n,entry in enumerate(summary['samples']):
            ident=entry['id'];row=rows[ident]
            for arm in (('base','sft') if n%2==0 else ('sft','base')):
                folder=output/'sources'/ident/arm;folder.mkdir(parents=True,exist_ok=False)
                backend.start_source(folder,arm);begin=time.monotonic()
                controller=build_program_controller(backend,study['registry'],study['rows'],study['budget'],skill_pool='F66')
                assert len(controller.library)==66
                trace=controller.run({'source_id':ident,'domain':row['domain'],'field_name':row['metadata']['field_name']})
                trace.write_json(folder/'trace.json')
                receipt={'id':ident,'arm':arm,'terminated':trace.terminated,'stop_reason':trace.stop_reason,
                    'calls':len(backend.calls),'backend_errors':len(backend.failures),'seconds':time.monotonic()-begin}
                put(folder/'receipt.json',receipt);entry['arms'][arm]=receipt
                print(a.cohort,ident,arm,trace.terminated,trace.stop_reason,flush=True)
                if backend.failures:raise RuntimeError('backend engineering failure; preserve all remaining denominator IDs')
            summary['paired_sources']+=1
            put(output/'progress'/('%04d.json'%(n+1)),summary)
        summary['status']='completed'
    finally:
        summary['seconds']=time.monotonic()-started
        put(output/'summary.json',summary)


if __name__=='__main__':main()
