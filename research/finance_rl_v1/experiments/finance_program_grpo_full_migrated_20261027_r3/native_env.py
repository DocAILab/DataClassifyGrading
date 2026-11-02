"""Instance-local instrumentation of exact native bindings and publications."""
from copy import deepcopy
import json
from types import SimpleNamespace
from dataflow import VersionedFlow,leaves
from reward import QueryLedger
from support import ROOT,VAL,load,native,digest


class Recorder:
    def __init__(self,source_version):
        self.flow=VersionedFlow();self.queries=QueryLedger();self.events=[]
        self.source_version=source_version;self.last={};self.terminal=None

    def install(self,ctrl):
        bridge=ctrl.bridge;executor=bridge.executor
        step=bridge.step;resolve=executor._resolved_arguments;current={}
        def captured(call,record,state):
            actual=resolve(call,record,state)
            current['resolved']=deepcopy(actual);current['canonical']=call.to_dict()
            return actual
        executor._resolved_arguments=captured
        def observe(call,record,state,counts,lineage,*args,**kwargs):
            current.clear();before=deepcopy(state);before_leaves=dict(leaves(before))
            # Unobserved controller-only changes are not attributed to a Skill.
            self.flow.invalidate([p for p in self.last if p not in before_leaves or self.last[p]!=before_leaves[p]])
            result=step(call,record,state,counts,lineage,*args,**kwargs)
            canonical=current.get('canonical') or (result.get('binding') or {}).get('normalized_call')
            reads=[]
            if canonical:
                for _,v in leaves(canonical.get('arguments',{})):
                    if isinstance(v,str) and v.startswith('$state.'):reads.append(v[7:])
            automatic=call.tool=='FINALIZE_CLASSIFICATION' and before.get('_provenance',{}).get('category')=='termination-first-v1'
            if automatic:
                # Exact frozen controller policy selects from these current candidates.
                reads.append('standard_candidates')
            after=dict(leaves(state));writes=[p for p,v in after.items() if p not in before_leaves or before_leaves[p]!=v]
            for key in result.get('data',{}):
                writes.extend(p for p in after if p==key or p.startswith(key+'.'))
            node=f's{len(self.events)+1:02d}'
            success=result.get('status')=='success';terminal=success and bool(bridge.catalog[call.tool].terminal)
            self.flow.add(node,sorted(set(reads)),sorted(set(writes)),success=success,terminal=terminal,automatic=automatic)
            self.flow.invalidate([p for p in before_leaves if p not in after])
            metadata_query=bridge.catalog[call.tool].role=='metadata_acquisition' and call.tool!='SELECT_CONTEXT_FIELDS'
            key=digest([record.source_id,call.tool,current.get('resolved',canonical or call.to_dict())])
            summary=result.get('summary','') or ''
            invalid=metadata_query and (self.queries.before(key,self.source_version) or
                'availability repeat target blocked' in summary or 'identical successful call blocked' in summary)
            if metadata_query:
                status='obtained' if success else ('missing' if 'Actual canonical metadata has no ' in summary or 'availability repeat target blocked' in summary else 'transient_error')
                self.queries.after(key,self.source_version,status)
            event=dict(node=node,tool=call.tool,status=result.get('status'),automatic=automatic,
                       captured_actual_binding='resolved' in current,canonical_call=deepcopy(canonical),
                       resolved_inputs_sha256=digest(current.get('resolved',{})),reads=sorted(set(reads)),
                       before_state_sha256=digest(before),after_state_sha256=digest(state),
                       invalid_query=int(invalid),query_key=key if metadata_query else None,
                       summary=summary,invalidations=deepcopy(result.get('invalidations',[])))
            self.events.append(event);self.last=after
            if terminal:self.terminal=node
            return result
        bridge.step=observe
        return ctrl

    def result(self,trace,guard_valid):
        result=self.flow.snapshot();result.update(events=deepcopy(self.events),terminal=self.terminal,
            invalid_count=sum(e['invalid_query'] for e in self.events),
            dependency_count=len(self.flow.rewarded(self.terminal)) if self.terminal else 0,
            native_guard_valid=guard_valid,source_version=self.source_version,
            dependency_scope='actual_explicit_bindings_and_publication_versions',
            semantic_dependency_proven=False,implicit_skill_prompt_augmentations_complete=False,
            dependency_training_eligible=False)
        return result


class Environment:
    def __init__(self,public):
        self.old=native();inputs=public['inputs']
        inputs=json.loads(json.dumps(inputs,ensure_ascii=False,sort_keys=True,separators=(',',':')))
        self.study=inputs['study'];self.cases=deepcopy(inputs['cases']);self.scopes=public['scopes']
        self.source_version=digest(self.study['rows']);engine=self.old.parent.load_engine(self.old.parent.E2)
        construct=engine.old.c.controller
        proxy=SimpleNamespace(**vars(engine));proxy.old=SimpleNamespace(**vars(engine.old));proxy.old.c=SimpleNamespace(**vars(engine.old.c))
        def create(*args,**kwargs):
            ctrl=construct(*args,**kwargs)
            if self.recorder is not None:self.recorder.install(ctrl)
            self.controller=ctrl;return ctrl
        proxy.old.c.controller=create
        exact=load('rl_accepted_native_runtime',VAL/'runtime.py')
        self.rollout=exact.build(proxy,self.study,self.scopes);self.recorder=None
        for case in self.cases:
            messages=[]
            def capture(m):messages.append(deepcopy(m));raise self.old.TargetReached()
            try:self.rollout(case['item'],self.study,capture,case['snapshot'])
            except self.old.TargetReached:pass
            if len(messages)!=1:raise ValueError('exact native initial target missing')
            case['messages']=messages[0]

    def run(self,case,generate):
        self.recorder=Recorder(self.source_version)
        trace=self.rollout(case['item'],self.study,generate,case['snapshot'])
        valid=trace.get('final_result') is not None and self.controller.bridge.guard._valid_terminal(trace['final_result'],trace['state'])
        audit=self.recorder.result(trace,bool(valid));self.recorder=None
        if len(audit['events'])!=trace['new_steps']:raise ValueError('every actual Skill attempt must be audited')
        return trace,audit
