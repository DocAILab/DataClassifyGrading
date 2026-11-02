"""Prospective budget-only adaptation of SHA-pinned native rollout, not old edits."""
import inspect

def limits_remaining(attempt,new_steps):
    return dict(programs=max(0,11-attempt),planner_attempts=max(0,11-attempt),
                skill_steps=max(0,32-new_steps),program_steps=max(0,min(16,32-new_steps)))

def build(engine):
    source=inspect.getsource(engine.rollout)
    replacements=[('max_programs=6, max_program_steps=12, max_skill_steps=16,\n                           max_planner_attempts=6',
                   'max_programs=10, max_program_steps=16, max_skill_steps=32,\n                           max_planner_attempts=10',1),
                  ('range(1, 7)','range(1, 11)',1),
                  ("trace['new_steps'] >= 16","trace['new_steps'] >= 32",2),
                  ('wide.limits_remaining(', 'limits_remaining(',2)]
    for old,new,count in replacements:
        if source.count(old)!=count:raise ValueError('unexpected frozen native source shape: '+old)
        source=source.replace(old,new)
    namespace=dict(engine.__dict__,limits_remaining=limits_remaining)
    exec(compile(source,'<pinned-native-budget16-32-10>','exec'),namespace)
    return namespace['rollout']
