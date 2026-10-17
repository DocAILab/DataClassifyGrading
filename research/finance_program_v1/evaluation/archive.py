"""After writer exit, snapshot all frozen code, logs, inputs, calls, routes and traces."""
import tarfile
from io_utils import ROOT,verify,put,read,sha


def main():
    verify();read(ROOT/'runs/exit.json')
    out=ROOT/'exports/final-01';out.mkdir(parents=True,exist_ok=False)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
        and 'exports' not in p.relative_to(ROOT).parts and p.relative_to(ROOT).as_posix()!='runs/watch.log')
    entries=[{'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in files]
    put(out/'members.json',entries)
    archive=out/'student-val80-evidence.tar.gz'
    with archive.open('xb') as stream,tarfile.open(fileobj=stream,mode='w:gz') as tar:
        for p in files:tar.add(p,arcname=p.relative_to(ROOT).as_posix(),recursive=False)
        tar.add(out/'members.json',arcname='archive-members.json',recursive=False)
    put(out/'archive.json',{'archive_sha256':sha(archive),'bytes':archive.stat().st_size,
        'members':len(entries)+1,'members_sha256':sha(out/'members.json'),'exit':read(ROOT/'runs/exit.json')})


if __name__=='__main__':main()
