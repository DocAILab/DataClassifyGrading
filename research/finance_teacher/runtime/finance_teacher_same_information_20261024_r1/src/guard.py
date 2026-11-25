"""Pure identity, one-claim and finite-campaign gates; no prompt augmentation."""
import copy,hashlib,json,math,os,re,time
from pathlib import Path

PRIVATE={'reference_label','expected_category','ground_truth','teacher_only_source_dossier','dossier','local_reference','grading_level','source_id','source_table_id'}

def canon(value):return json.dumps(value,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')
def sha(value):return hashlib.sha256(value).hexdigest()
def message_sha(messages):return sha(canon(messages))
def read(path):return json.loads(Path(path).read_bytes())
def put(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())

def strict_json(value):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out:raise ValueError('duplicate JSON key')
            out[k]=v
        return out
    def bad(_):raise ValueError('nonfinite JSON')
    return json.loads(value,object_pairs_hook=pairs,parse_constant=bad)

def forbidden_keys(value):
    if isinstance(value,dict):return (set(value)&PRIVATE)|set().union(*(forbidden_keys(v) for v in value.values()))
    if isinstance(value,list):return set().union(*(forbidden_keys(v) for v in value))
    return set()

def normalized_messages(messages):
    if not isinstance(messages,list) or not messages:raise ValueError('messages absent')
    out=[]
    for m in messages:
        if set(m)!={'role','content'} or m['role'] not in ('system','user','assistant','tool') or not isinstance(m['content'],str):raise ValueError('message envelope')
        if 'teacher_only_source_dossier' in m['content']:raise ValueError('teacher private append')
        out.append({'role':m['role'],'content':m['content']})
    if out[0]['role']!='system' or out[-1]['role']!='user':raise ValueError('unexpected role order')
    body=strict_json(out[-1]['content'])
    if forbidden_keys(body):raise ValueError('private target/identity in body')
    return out

def stage_of(messages):
    h=messages[0]['content'].splitlines()[0]
    match=re.fullmatch(r'Execute only the selected Skill (SELECT_CATEGORY_TOPK|INFER_CATEGORY_FROM_TOPK)\.',h)
    if not match:raise ValueError('not an approved Skill stage')
    return match[1]

def response_schema(messages):
    # Use the schema already included in the actual Student contract, unmodified.
    system=messages[0]['content']
    for match in re.finditer(r'\{"type":"object","additionalProperties":false,"required":',system):
        try:schema,_=json.JSONDecoder().raw_decode(system[match.start():])
        except ValueError:continue
        if {'option_id','summary','support'}<=set(schema.get('properties',{})) or {'option_ids','summary'}<=set(schema.get('properties',{})):return schema
    # Minimal equivalent for isolated unit fixtures without embedded schema.
    if stage_of(messages)=='SELECT_CATEGORY_TOPK':
        return dict(type='object',additionalProperties=False,required=['option_ids','summary'],properties=dict(option_ids=dict(type='array',items=dict(type='string')),summary=dict(type='string')))
    citation=dict(type='object',additionalProperties=False,required=['fact_id','quote'],properties=dict(fact_id=dict(type='string'),quote=dict(type='string')))
    support=dict(type='object',additionalProperties=False,required=['sufficient','citations','missing_evidence'],properties=dict(sufficient=dict(type='boolean'),citations=dict(type='array',maxItems=8,items=citation),missing_evidence=dict(type='string')))
    return dict(type='object',additionalProperties=False,required=['option_id','summary','support'],properties=dict(option_id=dict(type='string'),summary=dict(type='string'),support=support))

def build_body(messages,config):
    m=normalized_messages(messages)
    return dict(model=config['model'],messages=copy.deepcopy(m),temperature=config['temperature'],max_tokens=config['max_tokens'],thinking_budget=config['thinking_budget'],enable_thinking=True,enable_search=False,stream=True,stream_options={'include_usage':True},response_format=dict(type='json_schema',json_schema=dict(name='same_information_skill',strict=True,schema=response_schema(m))))

class Budget:
    def __init__(self,request_cap,cost_cap,deadline):
        self.request_cap=request_cap;self.cost_cap=cost_cap;self.deadline=deadline
        self.claims=0;self.spent=0.;self.pending=0.;self.halted=False
        self.input_tokens=0;self.output_tokens=0;self.unknown=0
    def admit(self,input_bytes,max_output):
        reserve=(input_bytes+4096)*12/1_000_000+max_output*36/1_000_000
        if self.halted or self.pending or self.claims>=self.request_cap or time.time()>=self.deadline or self.spent+reserve>self.cost_cap:raise ValueError('finite request/cost/deadline gate')
        self.claims+=1;self.pending=reserve;return reserve
    def debit(self,usage):
        valid=isinstance(usage,dict) and all(type(usage.get(k)) is int and usage[k]>=0 for k in ('prompt_tokens','completion_tokens','total_tokens')) and usage['total_tokens']==usage['prompt_tokens']+usage['completion_tokens']
        if not valid:
            self.spent+=self.pending;self.unknown+=1;self.halted=True
        else:
            cost=usage['prompt_tokens']*12/1_000_000+usage['completion_tokens']*36/1_000_000
            if cost>self.pending+1e-12:self.halted=True
            self.spent+=cost;self.input_tokens+=usage['prompt_tokens'];self.output_tokens+=usage['completion_tokens']
        self.pending=0.
    def snapshot(self):return {k:getattr(self,k) for k in ('request_cap','cost_cap','deadline','claims','spent','pending','halted','input_tokens','output_tokens','unknown')}

def claim_once(path,value):
    p=Path(path);p.mkdir(parents=True,exist_ok=False);put(p/'claim.json',value);return p

def check_expansion_gate(gate,freeze_sha,completion_sha):
    if not isinstance(gate,dict) or gate.get('passed') is not True or gate.get('freeze_sha256')!=freeze_sha or gate.get('pilot_completion_sha256')!=completion_sha or not re.fullmatch('[0-9a-f]{64}',gate.get('semantic_review_sha256','')) or not gate.get('criteria') or not all(v is True for v in gate['criteria'].values()):raise ValueError('independent bound semantic gate required')
    return True

def rollout_policy():return dict(max_skill_calls=None,max_executed_steps=None,campaign_budget_required=True,native_evidence_guards_unchanged=True,engine_validation_pending=True)
