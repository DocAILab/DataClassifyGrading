"""Original one-request wire child with suspend-aware bounded parent polling."""
import importlib.util,json,math,multiprocessing,sys,time
from pathlib import Path
_policy_path=Path(__file__).resolve().parent.parent/'finance_teacher_failedcases_retry_20261024_r9/recovery_policy.py'
_spec=importlib.util.spec_from_file_location('r10_recovery_policy',_policy_path);_policy=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(_policy)
ActiveWork,wait_seconds=_policy.ActiveWork,_policy.wait_seconds
BASE=Path(__file__).resolve().parent.parent/'finance_program_teacher_b150_20261020_r1/transport.py'
spec=importlib.util.spec_from_file_location('R6_wire_original',BASE);old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)

def child_args(folder,url,key):return (str(folder),url,str(Path(folder)/'request-body.bin'),key)

def _worker(folder,url,body_path,key):
 from r13_isolation import lease
 with lease(Path(__file__).resolve().parent/'active-http.lock'):old._capture_worker(folder,url,Path(body_path).read_bytes(),key)

def capture_sse(folder,url,body,key,model,timeout_seconds):
 if type(timeout_seconds) not in (int,float) or not math.isfinite(timeout_seconds) or timeout_seconds<=0:raise ValueError('positive finite HTTP timeout required')
 folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);(folder/'response.sse').open('xb').close()
 wall=time.time();mono=time.monotonic();wall_end=wall+timeout_seconds;mono_end=mono+timeout_seconds
 with (folder/'request-body.bin').open('xb') as request_file:request_file.write(body if isinstance(body,bytes) else json.dumps(body,ensure_ascii=False,separators=(',',':')).encode())
 child=multiprocessing.get_context('spawn').Process(target=_worker,args=child_args(folder,url,key));timed_out=False
 try:
  with ActiveWork():
   child.start()
   while child.is_alive():
    wait=wait_seconds(wall_end,mono_end,time.time(),time.monotonic())
    if wait<=0:timed_out=True;break
    child.join(wait)
   timed_out=timed_out or time.time()>=wall_end or time.monotonic()>=mono_end
   old._stop_child(child)
  elapsed_wall=time.time()-wall;elapsed_mono=time.monotonic()-mono
  life=dict(child_pid=child.pid,child_exitcode=child.exitcode,child_stopped=not child.is_alive(),timed_out=timed_out,seconds=elapsed_mono,wall_seconds=elapsed_wall,finite_wall_deadline=wall_end,suspend_or_scheduler_gap_possible=elapsed_wall>timeout_seconds+2,automatic_idle_sleep_request_scoped=True,explicit_lid_sleep_not_overridden=True,no_HTTP_retry=True)
  old._save(folder/'lifecycle.json',life)
  parser_spec=importlib.util.spec_from_file_location('R6_original_SSE_parser',old.PARSER_PATH);parser=importlib.util.module_from_spec(parser_spec);parser_spec.loader.exec_module(parser);result={}
  try:parser.parse_sse((folder/'response.sse').read_bytes().splitlines(),model,state=result)
  except Exception as exc:result['error_type']=type(exc).__name__
  status=None
  for name in ('http-headers.json','worker-status.json'):
   if (folder/name).exists():status=dict(status or {},**json.loads((folder/name).read_bytes()))
  if status is not None:
   for name in ('http_status','error_type','provider_error_code','failure_class','http_body_truncated','fatal_capture'):
    if name in status:result[name]=status[name]
  fatal=old.capture_failure(status,child.exitcode,timed_out);status=status or {};result['fatal_capture']=fatal
  if not status.get('done_event') or child.exitcode!=0 or status.get('error_type'):result.update(complete=False);result.setdefault('error_type','IncompleteCompletion')
  if timed_out:result.update(complete=False,error_type='TimeoutError')
  result['transport_complete']=bool(status.get('done_event') and child.exitcode==0 and not status.get('error_type') and not timed_out)
  result.update(timed_out=timed_out,child_pid=child.pid,child_exitcode=child.exitcode,child_stopped=not child.is_alive(),seconds=elapsed_mono)
  old._save(folder/'result.json',result);return result
 finally:
  if child.pid is not None:old._stop_child(child);child.close()
