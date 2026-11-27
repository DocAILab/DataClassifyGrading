from base_contracts import *
import re
PREFIX = '{"option_id":"'

class OptionInputTooLong(ValueError):
    """The intact ranking request exceeds its fixed input budget."""
    def __init__(self, count, limit, prompt_sha256):
        # Keep the existing message visible to the native recovery planner.
        super().__init__('no truncation permitted')
        self.evidence=dict(tokens=count,limit=limit,prompt_sha256=prompt_sha256)
def make_decision(case,registry,templates,ids):
    if len(ids) not in (5,12) or len(set(ids))!=len(ids):raise ValueError('fixed5 or full12 only')
    if len(ids)==12 and set(ids)!={x['option_id'] for x in registry}:raise ValueError('full12 registry')
    m=make_classification(case,registry,templates,ids[:5])
    u=json.loads(m[1]['content']);by={x['option_id']:x for x in registry}
    u['candidate_details']=[deepcopy(by[i]) for i in ids];m[1]['content']=dumps(u);return m

def encode_option(messages,option_id,tokenizer,max_length):
    if [x['role'] for x in messages]!=['system','user']:raise ValueError('no fabricated assistant messages')
    u=json.loads(messages[1]['content'])
    if not isinstance(option_id,str) or not re.fullmatch(r'C\d{3}',option_id):raise ValueError('provided concrete option ID required')
    if option_id not in {x['option_id'] for x in u['candidate_details']}:raise ValueError('target outside fixed catalog')
    prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    prefix=prompt+PREFIX;full=prefix+option_id
    enc=tokenizer(full,add_special_tokens=False,return_offsets_mapping=True)
    ids=enc['input_ids'];offsets=enc['offset_mapping'];start=len(prefix)
    if len(ids)>max_length:raise OptionInputTooLong(len(ids),max_length,__import__('hashlib').sha256(prompt.encode()).hexdigest())
    supervised=[i for i,(a,b) in enumerate(offsets) if b>start and a<len(full)]
    if not supervised or supervised!=list(range(supervised[0],len(ids))):raise ValueError('ID must be the contiguous supervised suffix')
    if offsets[supervised[0]][0]!=start or offsets[-1][1]!=len(full):raise ValueError('token boundary crosses JSON scaffolding')
    labels=[ids[i] if i in supervised else -100 for i in range(len(ids))]
    if tokenizer.eos_token_id in labels:raise ValueError('EOS must not be supervised')
    return {'input_ids':ids,'attention_mask':[1]*len(ids),'labels':labels},{'prompt_sha256':__import__('hashlib').sha256(prompt.encode()).hexdigest(),'supervised_text':option_id,'supervised_token_indices':supervised,'supervised_tokens':len(supervised),'total_tokens':len(ids),'assistant_scaffolding':PREFIX,'scaffolding_masked':True,'eos_appended':False,'eos_supervised':False,'summary_support_or_citation_supervised':False}

