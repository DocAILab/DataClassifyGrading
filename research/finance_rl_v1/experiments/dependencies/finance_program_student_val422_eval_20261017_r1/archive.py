"""After writer exit, snapshot all frozen code, logs, inputs, calls, routes and traces."""
import tarfile
from io_utils import ROOT,verify,put,read,sha


def main():
    verify();read(ROOT/'runs/exit.json')
    out=ROOT/'exports/final-01';out.mkdir(parents=True,exist_ok=False)
    # Only frozen upload inputs and generated run evidence; local-only reviews gold is never archived.
    frozen={x['path'] for x in read(ROOT/'manifest.json')['files']}|{'manifest.json'}
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and (p.relative_to(ROOT).as_posix() in frozen
             or p.relative_to(ROOT).parts[0] in ('runs','evidence')
             or p.relative_to(ROOT).as_posix() in ('reviews/cpu-checks.json','reviews/historical-proof.json',
                 'reviews/length-proof.json','reviews/val342-audit.json'))
        and p.relative_to(ROOT).as_posix()!='runs/watch.log')
    entries=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]
    put(out/'members.json',entries)
    archive=out/'student-val80-evidence.tar.gz'
    with archive.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        tar.add(out/'members.json',arcname='archive-members.json',recursive=False)
    put(out/'archive.json',{'archive_sha256':sha(archive),'bytes':archive.stat().st_size,
        'members':len(entries)+1,'members_sha256':sha(out/'members.json'),'exit':read(ROOT/'runs/exit.json'),
        'cohort':'val342','supplementary_expected_sources':342,'full_expected_sources':422,
        'evaluation_complete':False,'full422_pending_local_merge':True})


if __name__=='__main__':main()
