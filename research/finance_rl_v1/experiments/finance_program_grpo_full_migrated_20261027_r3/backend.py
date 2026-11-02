"""Original model engine, new sampled Planner and immutable reference weights."""
from copy import deepcopy
from contextlib import nullcontext
import json
import time
import types
from pathlib import Path
from execution_contract import Claims,options
from resource_runtime import publish_reference
from support import ROOT,TRAIN,ADAPTER_SHA,put,sha,digest,training


class ResourceFailure(BaseException):pass


def make_backend(admission):
    original=training();Base=original.ModelBackend
    if Base.__init__.__globals__['ROOT']!=TRAIN:raise ValueError('original model backend imported from wrong experiment root')
    class Backend(Base):
        def __init__(self):
            super().__init__(json.loads((TRAIN/'admission.json').read_bytes()))
            if self.expected['adapter_model.safetensors']!=ADAPTER_SHA:raise ValueError('original SFT identity differs')
            self.admission=admission;self.version=0;self.stochastic=True;self.stage=None
            self.claims=Claims(ROOT/'campaign/claims',admission['gpu_deadline_unix'],admission['model_call_limit'],carry_in=admission.get('inherited_claims',0))
            self.identity.update(RL_pilot=True,actor_initial_sha256=ADAPTER_SHA,policy_version=0,reference='frozen_final162',skill_route='base_only')
            namespace=dict(Base.generate.__globals__,generation_options=lambda a:options(self.stage,a['gpu_deadline_unix'],stochastic=self.stochastic))
            fn=types.FunctionType(Base.generate.__code__,namespace,Base.generate.__name__,Base.generate.__defaults__,Base.generate.__closure__)
            self.native_generate=types.MethodType(fn,self)
            self._prepared=False;self.behavior=None

        def _ensure_loaded(self):
            super()._ensure_loaded()
            if self._prepared:return
            torch=self.runtime['torch'];model=self.runtime['model']
            self.reference={n:p.detach().cpu().clone() for n,p in self.named}
            self.reference_hashes=dict(self.weight_shas)
            self.reference_digest=digest(self.reference_hashes)
            self.identity['reference_tensors_sha256']=self.reference_digest
            generate=model.generate
            def sampled(*args,**kwargs):
                if self.stage=='Planner' and self.stochastic:
                    answer=generate(*args,**kwargs,return_dict_in_generate=True,output_scores=True)
                    length=kwargs['input_ids'].shape[1];tokens=answer.sequences[0,length:]
                    if len(answer.scores)!=len(tokens):raise ValueError('sampled token/score alignment differs')
                    values=[float(torch.log_softmax(scores[0].float(),-1)[int(token.item())].item()) for scores,token in zip(answer.scores,tokens)]
                    self.behavior=dict(token_ids=tokens.detach().cpu().tolist(),log_probs=values,temperature=.8,
                                       top_p=1.,top_k=0,policy_version=self.version)
                    return answer.sequences
                return generate(*args,**kwargs)
            model.generate=sampled
            self._prepared=True
            publish_reference(ROOT/'runs/reference-identity.json',dict(initial_adapter_sha256=ADAPTER_SHA,tensors=self.reference_hashes,
                reference_digest=self.reference_digest,model_generation_config=model.generation_config.to_dict()))

        def resources(self):
            from resource_runtime import resources
            return resources(self)

        def generate(self,messages,arm,stage,destination):
            if arm!='N':raise ValueError('only Planner SFT/RL arm admitted')
            self.stage=stage;self.behavior=None
            self.last_generation_metadata={};self.last_call_evidence={}
            try:self.resources()
            except RuntimeError as exc:raise ResourceFailure(str(exc)) from exc
            claim=self.claims.claim('generate',destination,self.version)
            self.identity['policy_version']=self.version
            try:return self.native_generate(messages,arm,stage,destination)
            except Exception as exc:
                if 'out of memory' in str(exc).lower() or type(exc).__name__=='OutOfMemoryError':raise ResourceFailure(str(exc)) from exc
                raise
            finally:
                put(Path(destination)/'policy-evidence.json',dict(claim=claim,stage=stage,policy_version=self.version,
                    stochastic=stage=='Planner' and self.stochastic,behavior=self.behavior,
                    actor_tensor_digest=digest(self.weight_shas),reference_tensor_digest=getattr(self,'reference_digest',None)))

        def tensor_hashes(self):return {n:original.tensor_sha(p) for n,p in self.named}

        def copy_values(self,values):
            torch=self.runtime['torch']
            if set(values)!={n for n,_ in self.named}:raise ValueError('all80 weights required')
            with torch.no_grad():
                for n,p in self.named:p.copy_(values[n].to(device=p.device,dtype=p.dtype))

        def freeze_for_generation(self):
            model=self.runtime['model'];model.eval();model.config.use_cache=True
            for p in model.parameters():p.requires_grad_(False);p.grad=None
            self.weight_shas=self.tensor_hashes()

        def prepare_gradients(self):
            model=self.runtime['model'];torch=self.runtime['torch']
            from deps import configure_gradient_checkpointing
            if not getattr(self,'checkpointing_ready',False):
                proof=configure_gradient_checkpointing(model);model.enable_input_require_grads()
                from migration_contract import publish_configuration
                publish_configuration(ROOT/'runs/gradient-checkpointing.json',proof)
                put(ROOT/f'runs/gradient-checkpointing-attempt-{int(__import__("os").environ["FINANCE_ATTEMPT_ID"]):03d}.json',proof)
                self.checkpointing_ready=True
            model.train();model.config.use_cache=False
            names={n for n,_ in self.named}
            for n,p in model.named_parameters():p.requires_grad_(n in names)
            for module in model.modules():
                if isinstance(module,torch.nn.Dropout):module.eval()

        def cached_probability_replay(self,turn):
            """Forced old tokens reproduce the cached path; no new episode/gold."""
            torch=self.runtime['torch'];model=self.runtime['model'];self.freeze_for_generation()
            if self.version!=0:raise ValueError('diagnostic must precede all optimizer updates')
            resources=self.resources();index=self.claims.claim('actor_cached_replay_forward',turn['path'],self.version)
            old_stage=self.stage;self.stage='ProbabilityReplay';values=[];tokens=turn['tokens'];prompt=turn['prompt'];started=time.time()
            class ForceTokens:
                def __call__(_,input_ids,scores):
                    j=input_ids.shape[1]-len(prompt)
                    if not 0<=j<len(tokens):raise ValueError('forced replay token alignment differs')
                    values.append(float(torch.log_softmax(scores[0].float()/turn['temperature'],-1)[tokens[j]].item()))
                    scores.fill_(float('-inf'));scores[:,tokens[j]]=0.;return scores
            try:
                ids=torch.tensor([prompt],dtype=torch.long,device=model.get_input_embeddings().weight.device)
                kwargs=options('Planner',self.admission['gpu_deadline_unix'],stochastic=False)
                kwargs.update(max_new_tokens=len(tokens),logits_processor=[ForceTokens()],repetition_penalty=1.)
                route=self._state(False)
                device=ids.device;context=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
                with torch.inference_mode(),context:answer=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),**kwargs)
                actual=answer[0,len(prompt):].detach().cpu().tolist()
                if actual!=tokens or len(values)!=len(tokens):raise ValueError('forced cached replay differs from original actual tokens')
                delta=(torch.tensor(values)-torch.tensor(turn['old_log_probs'])).abs()
                proof=dict(kind='actor_cached_replay_forward',claim=index,turn=turn['path'],policy_version=0,
                    prompt_sha256=digest(prompt),action_sha256=digest(tokens),log_probs=values,action_tokens=len(tokens),
                    reference_tensor_digest=self.reference_digest,elapsed_seconds=time.time()-started,resources=resources,
                    max_abs_behavior_delta=float(delta.max()),mean_abs_behavior_delta=float(delta.mean()),
                    no_new_episode=True,temperature_applied_to_recorded_scores=turn['temperature'],actual_route=route)
                put(ROOT/f'runs/compute/{index:06d}.json',proof);put(ROOT/'runs/cached-probability-replay.json',proof)
                if proof['max_abs_behavior_delta']>.025 or proof['mean_abs_behavior_delta']>.001:
                    raise ValueError('cached forced replay did not reproduce the original behavior probabilities')
                return proof
            finally:self.stage=old_stage

        def log_probs(self,turn,*,reference=False,grad=False):
            from policy_math import action_log_probs
            torch=self.runtime['torch'];model=self.runtime['model']
            resources=self.resources();kind='reference_forward' if reference else ('actor_gradient_forward' if grad else 'actor_check_forward')
            index=self.claims.claim(kind,turn['path'],self.version);started=time.time()
            actor={n:p.detach().cpu().clone() for n,p in self.named} if reference else None
            if reference:self.copy_values(self.reference)
            try:
                if grad:self.prepare_gradients()
                else:model.eval();model.config.use_cache=False
                actual_digest=digest(self.tensor_hashes())
                expected_digest=self.reference_digest if reference else digest(self.weight_shas)
                if actual_digest!=expected_digest:raise ValueError('actual forward tensor identity differs')
                routes={n:dict(disabled=bool(m.disable_adapters),active_adapters=list(m.active_adapters)) for n,m in self.modules.items()}
                if len(routes)!=40 or any(m['disabled'] or m['active_adapters']!=['default'] for m in routes.values()):
                    raise ValueError('actual Planner forward adapter route differs')
                trainable=[n for n,p in model.named_parameters() if p.requires_grad]
                if grad and set(trainable)!={n for n,_ in self.named}:raise ValueError('gradient forward must train only80 Planner tensors')
                context=nullcontext() if grad else torch.no_grad()
                with context:values=action_log_probs(model,turn['prompt'],turn['tokens'],temperature=turn['temperature'])
                detached=values.detach().cpu();items=detached.tolist()
                put(ROOT/f'runs/compute/{index:06d}.json',dict(kind=kind,claim=index,turn=turn['path'],policy_version=self.version,
                    prompt_tokens=len(turn['prompt']),action_tokens=len(turn['tokens']),log_probs=items,
                    prompt_sha256=digest(turn['prompt']),action_sha256=digest(turn['tokens']),temperature=turn['temperature'],
                    actual_tensor_digest=actual_digest,actual_adapter_modules=routes,actual_trainable_names=trainable,
                    reference=reference,reference_tensor_digest=self.reference_digest,elapsed_seconds=time.time()-started,resources=resources))
                return values if grad else detached
            finally:
                if reference:self.copy_values(actor)

    return Backend()


def planner_turns(folder):
    turns=[]
    for p in sorted(Path(folder).glob('call-*/model-returned.json')):
        actual=json.loads(p.read_bytes())
        if actual['stage']!='Planner':continue
        policy=json.loads((p.parent/'policy-evidence.json').read_bytes())
        behavior=policy['behavior']
        if actual['error'] or actual['termination_reason']!='eos' or not behavior:continue
        if behavior['token_ids']!=actual['output_token_ids']:raise ValueError('actual sampled tokens differ')
        turns.append(dict(path=str(p.parent.relative_to(ROOT)),prompt=actual['prompt_token_ids'],tokens=actual['output_token_ids'],
                          temperature=behavior['temperature'],old_log_probs=behavior['log_probs'],policy_version=policy['policy_version']))
    return turns
