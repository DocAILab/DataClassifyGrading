"""Real CPU tests, original target identity and tokenizer; never load model."""
import io,json,time,unittest
from support import ROOT,TRAIN,N100,put,verify,load,native,training


def main():
    verify()
    import torch
    torch.set_num_threads(2)
    if torch.cuda.is_initialized():raise ValueError('CPU-only preflight')
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    put(ROOT/'reviews/remote-tests.json',dict(passed=tests.wasSuccessful(),tests=tests.testsRun,output=stream.getvalue()))
    if not tests.wasSuccessful():raise ValueError(stream.getvalue())
    from native_env import Environment
    from backend import make_backend
    public=json.loads((ROOT/'cohort-public.json').read_bytes());env=Environment(public);tok=training().tokenizer();lengths=[]
    interface=load('Planner_A1_preflight_interface',N100/'interface.py')
    for case in env.cases:
        probe=native().ContextBackend(case,None,'N',ROOT,probe=True)
        try:env.run(case,probe)
        except native().TargetReached:pass
        if not probe.target_exact or probe.records:raise ValueError('real initial target differs')
        messages=interface.RecoverySession().rewrite(case['messages'])
        text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        count=len(tok(text,add_special_tokens=False)['input_ids']);lengths.append(count)
        if count+512>49152:raise ValueError('no silent truncation')
    a=dict(json.loads((TRAIN/'admission.json').read_bytes()),gpu_deadline_unix=time.time()+600,model_call_limit=2000)
    backend=make_backend(a)
    if backend.runtime is not None or torch.cuda.is_initialized() or len(lengths)!=48:raise ValueError('CPU preflight initialized model/CUDA or cohort mismatch')
    if not (ROOT/'reward-labels.json').exists() or (ROOT/'evaluation-labels.json').exists():raise ValueError('training labels only')
    value=dict(passed=True,tests=tests.testsRun,targets=48,episodes=192,actual_initial_targets=48,
        input_tokens_min=min(lengths),input_tokens_max=max(lengths),model_calls=0,models_loaded=0,CUDA_initialized=False,rows_token_counts=lengths)
    put(ROOT/'reviews/preflight.json',value);print(json.dumps(value))


if __name__=='__main__':main()
