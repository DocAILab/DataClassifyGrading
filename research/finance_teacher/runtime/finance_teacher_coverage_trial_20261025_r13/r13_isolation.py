"""Real OS source isolation and exclusive worker/HTTP leases across restarts."""
import argparse,os,signal,subprocess,sys,time
if os.name == "nt":
 import msvcrt
else:
 import fcntl
from pathlib import Path
from contextlib import contextmanager
@contextmanager
def lease(path):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('a+b') as f:
  f.seek(0)
  if os.name == 'nt':msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
  else:fcntl.flock(f,fcntl.LOCK_EX | fcntl.LOCK_NB)
  try:yield
  finally:
   f.seek(0)
   if os.name == 'nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
   else:fcntl.flock(f,fcntl.LOCK_UN)
def run(cmd,target,deadline):
 target=Path(target)
 with (target/'stdout.log').open('xb') as out,(target/'stderr.log').open('xb') as err:
  w=subprocess.Popen(cmd,cwd=target,stdin=subprocess.DEVNULL,stdout=out,stderr=err,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0),start_new_session=os.name!="nt")
  from r13_common import put
  put(target/'launch.json',dict(pid=w.pid,parent_pid=os.getpid(),actual_cmd=cmd,time_unix=time.time()))
  forced=False
  while w.poll() is None:
   if time.time()>=deadline:
    forced=True
    if os.name!='nt':os.killpg(w.pid,signal.SIGKILL)
    else:subprocess.run(['taskkill','/PID',str(w.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10,check=False)
    w.wait(timeout=5);break
   time.sleep(.1)
 return dict(actual_OS_exit=w.returncode,worker_pid=w.pid,forced_finite_deadline=forced,time_unix=time.time())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--lease-probe',required=True);a=p.parse_args()
 try:
  with lease(a.lease_probe):pass
 except OSError:sys.exit(3)
