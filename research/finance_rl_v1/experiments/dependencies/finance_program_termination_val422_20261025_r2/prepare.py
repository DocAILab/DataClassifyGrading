import json
from full_policy import prepare
from support import ROOT,BASE,N100,TRAIN,VAL,put,sha


def main():
    study=json.loads((VAL/'data/study.json').read_bytes())
    original=json.loads((N100/'eval-inputs.json').read_bytes())['study']
    assert study['registry']==original['registry']
    train=json.loads((TRAIN/'data/data-lock.json').read_bytes())['identities']
    diagnostic=json.loads((N100/'selection.json').read_bytes())['slots']
    data=prepare(study,train,diagnostic)
    put(ROOT/'eval-inputs.json',dict(study=data['study'],cases=data['cases']))
    put(ROOT/'source-scopes.json',dict(rows=data['scopes']))
    put(ROOT/'selection.json',dict(slots=data['slots'],proof=data['proof'],model_calls_limit=17724,gpu_hours=8))
    put(ROOT/'reviews/data-proof.json',dict(data['proof'],source_study_sha256=sha(VAL/'data/study.json'),
                                          train_lock_sha256=sha(TRAIN/'data/data-lock.json'),
                                          exact_termination_sha256=sha(BASE/'termination.py'),exact_runtime_sha256=sha(BASE/'runtime.py')))
    print(json.dumps(data['proof']))


if __name__=='__main__':main()
