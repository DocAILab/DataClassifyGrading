"""Common Student/Teacher scope projection, never target-specific facts."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'finance_teacher_same_information_20261024_r2'
sys.path.insert(0,str(PARENT))
from revision import project_messages as parent_projection,read,put,sha,canon,message_sha,normalized_messages,binding,admit_revision as revision_admission
from src.guard import stage_of,build_body

def project_messages(messages,actor):
    out=parent_projection(messages,actor)
    if stage_of(out)=='SELECT_CATEGORY_TOPK':return out
    body=json.loads(out[-1]['content']);policy=read(ROOT/'scope-policy.json');rules=[]
    for candidate in body['candidate_details']:
        i=candidate['option_id']
        if i not in policy:raise ValueError('candidate not in frozen public scope policy')
        rules.append(dict(option_id=i,**policy[i]))
    out[0]['content']+='\nPublic definition scope rules (classification policy, not acquired facts):\n'+canon(rules).decode()+'\n逐项检查当前候选的正面必要范围。与例词相近或其他类不匹配，不替代正面范围证据。实际来源可用同义表述建立范围，不强制精确关键词。未满足的必要区分请明确不足；不要强制任何类别或强制不足。真实补证支持后仍可充分。summary请用简短中文，不超过20个字符。不得输出新键、schema、推理过程或围栏；其余原证据和JSON合同不变。'
    return normalized_messages(out)

def check_reuse(slot,request,outcome,config):
    if slot['stage']!='SELECT_CATEGORY_TOPK':raise ValueError('only unchanged SELECT reused')
    if request['body']!=build_body(slot['messages'],config) or message_sha(slot['messages'])!=slot['input_sha256'] or outcome['provider_target_sha256']!=slot['reuse_target_sha256'] or sha(outcome['provider_raw_content'].encode())!=slot['reuse_target_sha256']:raise ValueError('reuse message/body/target identity differs')
    return True
