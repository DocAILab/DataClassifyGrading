"""Keep native recovery/finalization unchanged, with an explicit planning limit."""
import inspect

from budget_transform import remaining_budget, rewrite_native_source, supported_limit
from recovery_runtime import RecoverySession


def build(engine, planner_attempt_limit):
    if not supported_limit(planner_attempt_limit):
        raise ValueError('only predeclared 10/20 planner limits are admitted')
    from support import N100, load, native
    native()
    from termination import finish
    previous = load('budget20_exact_n100_runtime', N100 / 'runtime.py')
    original = previous.build(engine)
    namespace = dict(original.__globals__)
    namespace['limits_remaining'] = lambda attempt, new_steps: remaining_budget(planner_attempt_limit, attempt, new_steps)
    namespace['finish'] = finish
    namespace['RecoverySession'] = RecoverySession
    source = rewrite_native_source(inspect.getsource(engine.rollout), planner_attempt_limit)
    changes=[
        ("                plan = parse_program(raw, ctrl.bridge.catalog, state, remaining['program_steps'])", "                plan = recovery.parse_program(raw, parse_program, ctrl.bridge.catalog, state, remaining['program_steps'])"),
        ("                feedback = dict(status='planner_rejected', error=attempt['error'], remaining_budget=remaining)", "                recovery.after_rejection(attempt['error'], backend_error=not isinstance(exc, ValueError))\n                feedback = dict(status='planner_rejected', error=attempt['error'], remaining_budget=remaining)"),
        ("            attempt['status'] = 'accepted'", "            attempt['status'] = 'accepted'\n            attempt['planner_recovery'] = deepcopy(recovery.last_plan_recovery)"),
        ('    guarded = boundary.Guard(generate)','    recovery = RecoverySession(generate)\n    guarded = boundary.Guard(recovery.generate)'),
        ('    ctrl = old.c.controller(guarded, local)','    ctrl = old.c.controller(guarded, local)\n    recovery.install(ctrl)'),
        ('            if guarded.halted:',"            if recovery.halted:\n                trace['stop_reason'] = recovery.stop_reason; break\n            if guarded.halted:"),
        ("                if result['status'] != 'success':\n                    obs.update(status='failed', failure_index=index); break", "                if result['status'] != 'success':\n                    if recovery.may_skip_redundant(result, state):\n                        result['recovery'] = {'action':'reused_existing_outputs','new_model_calls':0,'new_raw_fact_ids':[]}\n                        obs.setdefault('redundant_steps_skipped', []).append(index)\n                        continue\n                    obs.update(status='failed', failure_index=index); break"),
        ('            feedback = deepcopy(obs)','            recovery.after_program(obs)\n            feedback = deepcopy(obs)'),
        ('        return finish(trace, ctrl, record)',"        trace['recovery'] = recovery.snapshot()\n        if recovery.halted and not trace['terminated']:\n            trace.update(terminated=True, final_result=None, stop_reason='abstained_terminal',\n                termination={'policy':'bounded-contract-recovery-v1','outcome':'technical_abstention','original_stop_reason':recovery.stop_reason,'evidence_sufficient':False,'requires_review':True,'new_model_calls':0})\n            return trace\n        return finish(trace, ctrl, record)"),
    ]
    for before,after in changes:
        if source.count(before)!=1:raise ValueError('unexpected native recovery hook shape: '+before)
        source=source.replace(before,after)
    exec(compile(source, f'<native32-planner{planner_attempt_limit}>', 'exec'), namespace)
    return namespace['rollout']
