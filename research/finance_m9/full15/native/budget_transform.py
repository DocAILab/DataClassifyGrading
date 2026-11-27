"""Bounded budget-only adapter for the exact frozen native rollout."""


def supported_limit(value):
    return type(value) is int and value in (10, 20)


def rewrite_native_source(source, planner_attempt_limit):
    if not supported_limit(planner_attempt_limit):
        raise ValueError('unsupported planner attempt limit')
    replacements = [
        ('max_programs=6, max_program_steps=12, max_skill_steps=16,\n                           max_planner_attempts=6',
         f'max_programs={planner_attempt_limit}, max_program_steps=16, max_skill_steps=32,\n                           max_planner_attempts={planner_attempt_limit}', 1),
        ('range(1, 7)', f'range(1, {planner_attempt_limit + 1})', 1),
        ("trace['new_steps'] >= 16", "trace['new_steps'] >= 31", 2),
        ('wide.limits_remaining(', 'limits_remaining(', 2),
        ('        return trace', '        return finish(trace, ctrl, record)', 1),
    ]
    for before, after, count in replacements:
        if source.count(before) != count:
            raise ValueError('unexpected frozen native source shape: ' + before)
        source = source.replace(before, after)
    return source


def remaining_budget(planner_attempt_limit, attempt, new_steps):
    if not supported_limit(planner_attempt_limit):
        raise ValueError('unsupported planner attempt limit')
    return dict(programs=max(0, planner_attempt_limit+1-attempt), planner_attempts=max(0, planner_attempt_limit+1-attempt),
                skill_steps=max(0, 32-new_steps), program_steps=max(0, min(16, 32-new_steps)))
