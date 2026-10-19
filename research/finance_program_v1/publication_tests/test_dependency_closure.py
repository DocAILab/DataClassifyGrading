"""Fresh-process public entrypoint checks; no private workspace or model calls."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicDependencyTests(unittest.TestCase):
    def run_python(self, folder, code):
        prefix = 'import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd())); '
        result = subprocess.run([sys.executable, '-I', '-B', '-c', prefix + code],
                                cwd=ROOT / folder, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nontraining_audit_imports_without_private_checkout(self):
        self.run_python('nontraining',
            'import batch_audit; from reviews.independent_audit_01 import RecordedBackend; '
            'b=RecordedBackend([]); b.finish(); '
            'assert "torch" not in sys.modules and "transformers" not in sys.modules')

    def test_pool_metadata_resolves_through_public_controller(self):
        self.run_python('nontraining',
            'import program_loop; import pool_metadata; '
            'assert Path(pool_metadata.__file__).resolve().parent == Path.cwd()/"vendor"')

    def test_teacher_recorded_audit_and_runtime_have_local_code(self):
        self.run_python('teacher',
            'import detached, pilot_audit, runtime_identity; '
            'assert callable(pilot_audit.replay_retry_one); '
            'assert Path(runtime_identity.__file__).resolve().is_relative_to(Path.cwd().parent); '
            'assert "torch" not in sys.modules')

    def test_legacy_revision_helper_has_no_private_import(self):
        self.run_python('evaluation',
            'import io_utils; from tools.freeze_revision import safe_path, signature; '
            'assert safe_path(Path.cwd(),"safe/file.json")==Path.cwd()/"safe/file.json"; '
            'assert len(signature([]))==64')

    def test_teacher_nonterminal_trace_replays_and_rejects_tampering(self):
        self.run_python('teacher', '''
import json, copy
import detached
from program_loop import build_program_controller
from teacher_api import api_messages,digest,parse_sse,role
from pilot_audit import replay_retry_one
study={'rows':[{'id':'fixture-001','domain':'finance','metadata':{'field_name':'balance','field_description':'fixture description'}}],'registry':{'cat'+str(i):{'description':'fixture class '+str(i),'grading_level':'L1'} for i in range(1,13)},'budget':{'max_programs':3,'max_program_steps':10,'max_skill_steps':12,'max_planner_attempts':4,'max_identical_calls':1}}
c=build_program_controller(lambda m:'',study['registry'],study['rows'],study['budget'],skill_pool='F66')
allowed=[s['name'] for s in c.library]; records=[]; logical=[]
def backend(messages):
    system=messages[0]['content']
    if 'Global Program Planner' in system:
        value={'program':[{'tool':n,'arguments':{}} for n in ('GET_FIELD_METADATA','INTERPRET_FIELD_NAME','SELECT_CATEGORY_TOPK','INFER_CATEGORY_FROM_TOPK','INFER_GRADING_LEVEL','FINALIZE_CLASSIFICATION')]}
    elif 'selected Skill SELECT_CATEGORY_TOPK.' in system: value={'option_ids':['C001','C002','C003','C004','C005'],'summary':'rank'}
    elif 'selected Skill INFER_CATEGORY_FROM_TOPK.' in system: value={'option_id':'C001','summary':'choice'}
    else: value={'summary':'interpret','data':{'field_semantics':'observed field'}}
    raw=json.dumps(value)
    body={'model':'qwen3.8-2.4t-a95b','messages':api_messages(messages),'response_format':{'type':'json_object'}}
    q={'body':body,'body_sha256':digest(body),'executor_messages':messages,'role':role(messages,allowed),'credential_recorded':False}
    events=[{'model':body['model'],'choices':[{'index':0,'delta':{'content':raw},'finish_reason':'stop'}],'usage':{'prompt_tokens':1,'completion_tokens':2,'total_tokens':3}}]
    x=parse_sse(['data: '+json.dumps(e) for e in events]+['data: [DONE]'],body['model'])
    x['response_sha256']=digest({'content':x['content'],'reasoning_content':x['reasoning_content']})
    n=len(records); records.append({'request':q,'result':x})
    logical.append({'executor_messages_sha256':digest(messages),'call_start':n,'call_end':n+1,'complete':True,'error':None})
    return raw
c=build_program_controller(backend,study['registry'],study['rows'],study['budget'],skill_pool='F66')
row=study['rows'][0]; saved=c.run({'source_id':row['id'],'domain':row['domain'],'field_name':row['metadata']['field_name']}).to_dict()
proof=replay_retry_one(study,row,saved,records,logical,allowed)
assert proof['full_trace_exact'] and proof['all_calls_consumed'] and proof['api_attempts']==4
assert not proof['terminated'] and proof['stop_reason']=='program_budget_exhausted'
changed=copy.deepcopy(records); changed[0]['request']['body_sha256']='0'*64
try: replay_retry_one(study,row,saved,changed,logical,allowed)
except AssertionError: pass
else: raise AssertionError('tampered teacher request accepted')
assert 'torch' not in sys.modules
''')

    def test_public_audit_manifest_binds_shipped_source(self):
        self.run_python('nontraining',
            'from local_smoke import environment; from reviews.independent_audit_01 import validate_sources; '
            'validate_sources(Path.cwd(),environment("not-a-model"))')

    def test_public_source_audit_rejects_mutation_and_missing_file(self):
        self.run_python('nontraining', '''
import shutil, tempfile
from local_smoke import environment
from reviews.independent_audit_01 import validate_sources, AuditError
with tempfile.TemporaryDirectory() as d:
    snapshot=Path(d)/'snapshot'; shutil.copytree(Path.cwd(),snapshot)
    environment_evidence=environment('not-a-model')
    validate_sources(snapshot,environment_evidence)
    target=snapshot/'vendor/pool_metadata.py'; original=target.read_bytes()
    target.write_bytes(original+bytes([10])+b'# changed')
    try: validate_sources(snapshot,environment_evidence)
    except AuditError: pass
    else: raise AssertionError('changed source accepted')
    target.unlink()
    try: validate_sources(snapshot,environment_evidence)
    except AuditError: pass
    else: raise AssertionError('missing source accepted')
''')

    def test_batch_audit_replays_real_scripted_controller(self):
        self.run_python('nontraining', '''
import json, tempfile, hashlib
from batch_runner import run_batch
from batch_audit import audit
class Scripted:
    def __init__(self, model, call_dir):
        self.folder=Path(call_dir); self.folder.mkdir(parents=True); self.calls=[]; self.failures=[]
    def __call__(self,messages):
        system=messages[0]['content']
        if 'Global Program Planner' in system:
            raw=json.dumps({'program':[{'tool':n,'arguments':{}} for n in ('GET_FIELD_METADATA','INTERPRET_FIELD_NAME','SELECT_CATEGORY_TOPK','INFER_CATEGORY_FROM_TOPK','INFER_GRADING_LEVEL','FINALIZE_CLASSIFICATION')]})
        elif 'selected Skill SELECT_CATEGORY_TOPK.' in system:
            raw=json.dumps({'option_ids':['C001','C002','C003','C004','C005'],'summary':'rank'})
        elif 'selected Skill INFER_CATEGORY_FROM_TOPK.' in system:
            raw=json.dumps({'option_id':'C001','summary':'choice'})
        else: raw=json.dumps({'summary':'interpret','data':{'field_semantics':'observed field'}})
        folder=self.folder/('call-%04d'%(len(self.calls)+1)); folder.mkdir(); self.calls.append(str(folder))
        for name,value in [('request',{'messages':messages}),('result',{'messages':messages,'raw_text':raw,'error':None,'input_tokens':1,'output_tokens':2,'seconds':0.01})]:
            (folder/(name+'.json')).write_text(json.dumps(value),encoding='utf-8')
        return raw
with tempfile.TemporaryDirectory() as d:
    d=Path(d)
    study={'schema':1,'cohorts':{'smoke':['fixture-001'],'pilot':[],'expanded':[]},'rows':[{'id':'fixture-001','domain':'finance','metadata':{'field_name':'balance','field_description':'fixture description'}}],'registry':{'cat'+str(i):{'description':'fixture class '+str(i),'grading_level':'L1'} for i in range(1,13)},'budget':{'max_programs':3,'max_program_steps':10,'max_skill_steps':12,'max_planner_attempts':4,'max_identical_calls':1}}
    p=d/'study.json'; p.write_text(json.dumps(study),encoding='utf-8')
    (d/'study-manifest.json').write_text(json.dumps({'study_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}),encoding='utf-8')
    assert run_batch('not-a-model',d/'run','smoke',study_path=p,backend_factory=Scripted)==0
    proof=audit(d/'run',study_path=p,allow_scripted_fixture=True)
    assert proof['verified_samples']==1 and proof['full_trace_exact'] and proof['all_calls_consumed']
    q=d/'run/calls/call-0001/request.json'; data=json.loads(q.read_bytes()); data['messages'][0]['content']+='changed'; q.write_text(json.dumps(data),encoding='utf-8')
    try: audit(d/'run',study_path=p,allow_scripted_fixture=True)
    except AssertionError: pass
    else: raise AssertionError('tampered request incorrectly accepted')
''')


if __name__ == '__main__':
    unittest.main()
