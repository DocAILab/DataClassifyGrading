"""Fresh-process public entrypoint checks; no private workspace or model calls."""

from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicDependencyTests(unittest.TestCase):
    def run_python(self, folder, code):
        prefix = "import sys; from pathlib import Path; sys.path.insert(0,str(Path.cwd())); "
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", prefix + code],
            cwd=ROOT / folder,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_mainline_uses_one_complete_f66_runtime(self):
        self.run_python(
            "evaluation",
            """
import io_utils
from program_loop import build_program_controller
import pool_metadata
registry={'fixture-'+str(i): {'description':'synthetic class', 'grading_level':'L1'} for i in range(12)}
budget={'max_programs':3,'max_program_steps':10,'max_skill_steps':12,'max_planner_attempts':4,'max_identical_calls':1}
controller=build_program_controller(lambda messages:'invalid JSON',registry,[],budget,skill_pool='F66')
assert len(controller.library)==66
assert Path(pool_metadata.__file__).resolve()==Path.cwd()/'runtime/vendor/pool_metadata.py'
assert not (Path.cwd().parent/'nontraining/program_loop/controller.py').exists()
trace=controller.run({'domain':'finance','field_name':'synthetic-field'})
assert not trace.terminated and trace.stop_reason=='planner_budget_exhausted'
assert trace.budget=={'programs':0,'planner_attempts':4,'skill_steps':0}
assert 'torch' not in sys.modules and 'transformers' not in sys.modules
""",
        )

    def test_teacher_recorded_audit_and_runtime_have_local_code(self):
        self.run_python(
            "teacher",
            "import detached, pilot_audit, runtime_identity; "
            "assert callable(pilot_audit.replay_retry_one); "
            "assert Path(runtime_identity.__file__).resolve().is_relative_to(Path.cwd().parent); "
            'assert "torch" not in sys.modules',
        )

    def test_legacy_revision_helper_has_no_private_import(self):
        self.run_python(
            "evaluation",
            "import io_utils; from tools.freeze_revision import safe_path, signature; "
            'assert safe_path(Path.cwd(),"safe/file.json")==Path.cwd()/"safe/file.json"; '
            "assert len(signature([]))==64",
        )

    def test_teacher_nonterminal_trace_replays_and_rejects_tampering(self):
        self.run_python(
            "teacher",
            """
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
""",
        )


if __name__ == "__main__":
    unittest.main()
