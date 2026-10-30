"""New-window preflight: native targets and original tokenizer, zero model generation."""
import json,sys,hashlib,time
from pathlib import Path
from common import ROOT,PARENT
import interface
from policy import gate_slots
sys.path.append(str(PARENT))
from eval_runtime import prepare,preflight as native_preflight
from token_audit import tokenizer
def run():
    from backend import EvaluationBackend
    original=json.loads((PARENT/'admission.json').read_bytes())
    evaluation=dict(original,evaluation_only=True,gpu_started_unix=time.time(),gpu_deadline_unix=time.time()+8*3600)
    backend=EvaluationBackend(evaluation)
    assert backend.runtime is None and backend.admission==evaluation and len(backend.expected_tensors)==80
    native=native_preflight()
    slots=json.loads((PARENT/'selection.json').read_bytes())['slots'];chosen=gate_slots(slots)
    study,cases,rollout=prepare();ids={s['source_id'] for s in chosen};tok=tokenizer();lengths=[]
    for case in cases:
        if case['source_id'] not in ids:continue
        actual=interface.RecoverySession().rewrite(case['messages'])
        text=tok.apply_chat_template(actual,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        n=len(tok(text,add_special_tokens=False)['input_ids']);assert n+512<=49152
        lengths.append(dict(source_id=case['source_id'],tokens=n,sha256=hashlib.sha256(text.encode()).hexdigest()))
    assert len(lengths)==20
    return dict(native=native,gate_tokens=lengths,new_model_calls=0,no_truncation=True,
        separate_evaluation_admission_accepted=True,expected_fp32_tensors=len(backend.expected_tensors))
if __name__=='__main__':print(json.dumps(run()))
