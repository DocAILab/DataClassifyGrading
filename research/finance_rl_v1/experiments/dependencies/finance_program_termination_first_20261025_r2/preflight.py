"""CPU-only exact initial requests and real pinned tokenizer checks for all100."""
import json
import sys
import time
from support import ROOT, N100, TRAIN, load, native, put, verify


def run():
    verify()
    old = native()
    study,cases,rollout = load('termination_first_preflight_runtime',ROOT/'runtime.py').prepare()
    for case in cases:
        bridge = old.ContextBackend(case,None,'N',ROOT,probe=True)
        try:
            rollout(case['item'],study,bridge,case['snapshot'])
        except old.TargetReached:
            pass
        assert bridge.target_exact and not bridge.records
    sys.path.append(str(TRAIN))
    from token_audit import tokenizer
    interface = load('termination_first_preflight_interface',N100/'interface.py')
    tok = tokenizer()
    tokens = []
    for case in cases:
        messages = interface.RecoverySession().rewrite(case['messages'])
        prompt = tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        count = len(tok(prompt,add_special_tokens=False)['input_ids'])
        assert count+512 <= 49152
        tokens.append(count)
    from live_backend import EvaluationBackend
    admission = json.loads((TRAIN/'admission.json').read_bytes())
    admission.update(evaluation_only=True,gpu_deadline_unix=time.time()+3600)
    backend = EvaluationBackend(admission)
    assert backend.runtime is None and len(backend.expected_tensors)==80
    # All-malformed planners still close in exactly10 model attempts.
    calls=[]
    def malformed(messages):
        calls.append(messages)
        return '{}'
    trace=rollout(cases[0]['item'],study,malformed,cases[0]['snapshot'])
    assert trace['terminated'] and trace['stop_reason']=='abstained_terminal' and len(calls)==10
    proof=dict(passed=True,exact_initial_targets=len(cases),new_model_calls=0,gpu_loaded=False,
               initial_tokens_min=min(tokens),initial_tokens_max=max(tokens),
               expected_fp32_tensors=80,N_only=True,malformed_planner_bounded=True)
    put(ROOT/'reviews/preflight.json',proof)
    return proof


if __name__=='__main__':
    print(json.dumps(run()))
