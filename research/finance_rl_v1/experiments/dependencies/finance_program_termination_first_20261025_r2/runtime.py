"""Native recovery with one reserved terminal step and controller-owned fallback."""
import inspect
from support import N100, load, native
from termination import finish


def build(engine):
    native()  # Ensure the frozen runtime dependencies use their original roots.
    previous = load('termination_first_n100_runtime', N100 / 'runtime.py')
    original = previous.build(engine)
    namespace = dict(original.__globals__)
    source = inspect.getsource(engine.rollout)
    replacements = [
        ('max_programs=6, max_program_steps=12, max_skill_steps=16,\n                           max_planner_attempts=6',
         'max_programs=10, max_program_steps=16, max_skill_steps=32,\n                           max_planner_attempts=10', 1),
        ('range(1, 7)', 'range(1, 11)', 1),
        ("trace['new_steps'] >= 16", "trace['new_steps'] >= 31", 2),
        ('wide.limits_remaining(', 'limits_remaining(', 2),
        ('        return trace', '        return finish(trace, ctrl, record)', 1),
    ]
    for before, after, count in replacements:
        if source.count(before) != count:
            raise ValueError('frozen native source shape differs: ' + before)
        source = source.replace(before, after)
    namespace['finish'] = finish
    exec(compile(source, '<termination-first-native16-32-10>', 'exec'), namespace)
    return namespace['rollout']


def prepare():
    old = native()
    study, cases, _ = old.prepare()
    engine = old.parent.load_engine(old.parent.E2)
    return study, cases, build(engine)
