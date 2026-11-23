"""Instance-only format feedback and transparent reuse of already published outputs."""
from copy import deepcopy
import hashlib,json,re
from prompt200 import format_messages

def digest(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def stage_of(messages):
    first=messages[0]['content'].splitlines()[0]
    return first.removeprefix('Execute only the selected Skill ').removesuffix('.') if first.startswith('Execute only the selected Skill ') else 'Planner'

class RecoverySession:
    def __init__(self,generate):
        self.backend=generate;self.failed={};self.requests={};self.last_request={};self.success={}
        self.current_state={}
        self.last_plan_recovery=None;self.last_planner_validation_error=None
        self.recent=[];self.events=[];self.idle_programs=0;self.halted=False;self.stop_reason=None

    def completion_bindings(self,payload):
        """Report only actual state presence and frozen dependency contracts, no plans."""
        rules=payload.get('shared_input_rules',{}).get('final_evidence_state',{})
        required=rules.get('requires_all',[])
        if not isinstance(required,list) or not all(isinstance(x,str) for x in required):required=[]
        missing=[x for x in required if x not in self.current_state]
        producers=[]
        for skill in payload.get('skill_library',[]):
            produced=set(skill.get('produces',[]))|set(skill.get('planning',{}).get('conditional_outputs',[]))
            needed=sorted(produced&set(missing))
            if needed:producers.append({'tool':skill['name'],'missing_outputs_it_may_produce':needed,'missing_required_inputs':[x for x in skill.get('required_inputs',[]) if x not in self.current_state]})
        return {'required_prerequisites':required,'missing_prerequisites':missing,'all_prerequisite_bindings_present':bool(required) and not missing,
            'classification_completed':'final_result' in self.current_state,'possible_producers':producers,
            'instruction':'These are state-presence and frozen-contract observations, not proof of correctness or an executable plan. A category-only intermediate result is not completion. Retain current valid outputs, choose the missing dependency work, and finish through native terminal validation; no step is inserted automatically. Repeating the same classification does not produce a missing grading binding.'}

    def generate(self,messages):
        actual=format_messages(messages);stage=stage_of(actual)
        if stage=='Planner':
            payload=json.loads(actual[-1]['content'])
            payload['recovery_context']={'completed_outputs':[{'tool':k,'output_keys':sorted(v)} for k,v in sorted(self.success.items()) if all(key in self.current_state and self.current_state[key]==value for key,value in v.items())],
                'completion_bindings':self.completion_bindings(payload),
                'last_planner_validation_error':self.last_planner_validation_error,
                'recent_failures':deepcopy(self.recent[-6:]),'instruction':'Reuse existing published state. Do not restart completed tools. Avoid a previously blocked prefix; continue with legal unexecuted work. Validator errors are not new business evidence.'}
            actual[-1]['content']=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        else:
            key=digest(actual);self.last_request[stage]=key
            if key in self.failed:
                if self.requests.get(key,0)>=2:
                    self.events.append({'kind':'repeated_invalid_output_blocked','stage':stage,'request_sha256':key,'new_model_calls':0})
                    raise ValueError('recovery_repeated_invalid_output_blocked: same actual inputs already had one validation repair')
                payload=json.loads(actual[-1]['content'])
                payload['repair_feedback']={'validator_error':self.failed[key], 'instruction':'Correct only the returned JSON/schema/type or invalid citation using the existing evidence_boundary. No new facts or guessed support. If actual evidence is insufficient follow the existing UNRESOLVED contract.'}
                actual[-1]['content']=json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':'))
                self.events.append({'kind':'validation_repair_requested','stage':stage,'request_sha256':key})
            self.requests[key]=self.requests.get(key,0)+1
        return self.backend(actual)

    def note_failure(self,stage,error):
        error=str(error)[:600];self.recent.append({'tool':stage,'validator_error':error})
        key=self.last_request.get(stage)
        repairable=any(x in error for x in ('invalid JSON','support_schema','declared evidence type','exact classification keys','citation:','quote:'))
        if key and repairable and 'insufficient:' not in error:self.failed[key]=error

    def note_success(self,stage,data):
        if data:self.success[stage]=deepcopy(data);self.current_state.update(deepcopy(data))

    def install(self,ctrl):
        previous=ctrl.bridge.step
        def step(call,record,state,counts,lineage,*args,**kwargs):
            result=previous(call,record,state,counts,lineage,*args,**kwargs)
            self.current_state=deepcopy(state)
            if result.get('status')=='success':self.note_success(call.tool,result.get('data',{}))
            else:self.note_failure(call.tool,result.get('summary',''))
            return result
        ctrl.bridge.step=step

    def may_skip_redundant(self,result,state):
        stage=result.get('call',{}).get('tool')
        if stage in ('FINALIZE_CLASSIFICATION','CONFIRM_GRADING_LEVEL','INFER_CATEGORY','INFER_CATEGORY_FROM_TOPK'):return False
        previous=self.success.get(stage)
        return bool(result.get('status')=='blocked' and result.get('summary')=='identical successful call blocked for current evidence lineage'
            and previous and all(k in state and state[k]==v for k,v in previous.items()))

    def after_program(self,obs):
        advanced=any(x.get('status')=='success' for x in obs['steps'])
        self.idle_programs=0 if advanced else self.idle_programs+1
        if self.idle_programs>=3:self.halted=True;self.stop_reason='recovery_no_progress'

    def parse_program(self,raw,original_parse,catalog,state,max_steps):
        self.last_plan_recovery=None
        try:return original_parse(raw,catalog,state,max_steps)
        except ValueError as exc:
            match=re.fullmatch(r'step (\d+): missing dependency (\w+)',str(exc))
            if not match or int(match[1])==0:raise
            # The full original parser already accepted JSON uniqueness/shape/bounds,
            # and all earlier dependencies. Revalidate the exact prefix independently.
            value=json.loads(raw);n=int(match[1]);prefix={'program':value['program'][:n]}
            parsed=original_parse(json.dumps(prefix),catalog,state,max_steps)
            if not parsed.steps or any(catalog[x.tool].terminal for x in parsed.steps):raise
            self.last_plan_recovery={'policy':'original-validator-legal-prefix-v1','original_validation_error':str(exc),'validated_prefix_length':n,'original_program_length':len(value['program']),'original_response_sha256':hashlib.sha256(raw.encode()).hexdigest(),'unexecuted_original_tail':deepcopy(value['program'][n:]),'new_business_facts':0}
            self.events.append(deepcopy(self.last_plan_recovery))
            return parsed

    def after_rejection(self,error,backend_error=False):
        self.last_planner_validation_error=str(error)[:600]
        if backend_error:self.halted=True;self.stop_reason='model_backend_failure'
        else:self.after_program({'steps':[]})

    def snapshot(self):
        return {'protocol':'bounded-contract-recovery-v1','events':deepcopy(self.events),'consecutive_no_progress_programs':self.idle_programs,
                'halted':self.halted,'stop_reason':self.stop_reason,'new_business_facts_supplied_by_repair':0}
