"""Actual full422 initial requests/tokenizer; no weights or GPU loaded."""
import json,sys,time
from support import ROOT,TRAIN,N100,BASE,load,native,put,sha,verify


def run():
    verify();old=native()
    study,cases,rollout=load('full422_preflight_runtime',ROOT/'runtime.py').prepare()
    for case in cases:
        bridge=old.ContextBackend(case,None,'N',ROOT,probe=True)
        try:rollout(case['item'],study,bridge,case['snapshot'])
        except old.TargetReached:pass
        assert bridge.target_exact and not bridge.records
    sys.path.append(str(TRAIN))
    from token_audit import tokenizer
    tok=tokenizer();interface=load('full422_preflight_interface',N100/'interface.py');tokens=[]
    for case in cases:
        m=interface.RecoverySession().rewrite(case['messages'])
        text=tok.apply_chat_template(m,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        count=len(tok(text,add_special_tokens=False)['input_ids']);assert count+512<=49152;tokens.append(count)
    from live_backend import EvaluationBackend
    admission=json.loads((TRAIN/'admission.json').read_bytes());admission.update(evaluation_only=True,gpu_deadline_unix=time.time()+3600,scientific_updates_allowed=0,maximum_optimizer_updates=0)
    backend=EvaluationBackend(admission);assert backend.runtime is None and len(backend.expected_tensors)==80
    proof=dict(passed=True,targets=len(cases),model_calls=0,exact_initial_targets=len(cases),new_model_calls=0,gpu_loaded=False,expected_fp32_tensors=80,
               initial_tokens_min=min(tokens),initial_tokens_max=max(tokens),N_only=True,
               exact_termination_sha256=sha(BASE/'termination.py'),exact_runtime_sha256=sha(BASE/'runtime.py'))
    put(ROOT/'reviews/preflight.json',proof);return proof


if __name__=='__main__':print(json.dumps(run()))
