"""One fixed stage, serial real API calls; no retries or Teacher-only context."""
import argparse,importlib.util,json,os,re,sys,time
from pathlib import Path
from src.guard import Budget,build_body,canon,sha,message_sha,normalized_messages,read,put,claim_once,check_expansion_gate
ROOT=Path(__file__).resolve().parent
TRANSPORT=ROOT.parent/'finance_program_teacher_b150_20261020_r1/transport.py'
PARSER=ROOT.parent/'finance_program_teacher_balanced_20261016_r9/lib/teacher_api.py'
CREDENTIALS=None
RULES=ROOT.parent/'finance_skill_supervision_20261024_r1/src/rules.py'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def verify_bindings(bindings):
    for b in bindings:
        raw=Path(b['path']).read_bytes()
        if len(raw)!=b['bytes'] or sha(raw)!=b['sha256']:raise ValueError('frozen dependency changed')

def transport_critical_ok(r,model):
    return bool(r.get('http_status')==200 and r.get('transport_complete') is True and r.get('child_stopped') is True and r.get('child_exitcode')==0 and r.get('timed_out') is False and r.get('models')==[model] and not r.get('fatal_capture'))

