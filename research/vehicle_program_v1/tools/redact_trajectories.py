"""Allowlisted lossy projection; not a training set or full replay artifact."""
import hashlib
import json
import re


def public_trace(trace, public_id, trace_sha256, receipt_status, allowed_tools, category_aliases, grade_aliases):
    if not re.fullmatch(r'val-\d{4}', public_id) or not re.fullmatch(r'[0-9a-f]{64}', trace_sha256):
        raise ValueError('invalid public identity')
    statuses = {'accepted', 'rejected', 'error', 'blocked', 'success', 'completed', 'terminated', 'backend_error', 'interrupted'}
    stops = {'validated_terminal', 'program_budget_exhausted', 'planner_budget_exhausted', 'skill_budget_exhausted'}
    def status(value):
        return value if isinstance(value, str) and value in statuses else 'other'
    def number(value):
        return value if type(value) is int and 0 <= value <= 10000 else None
    def call(value):
        value = value if isinstance(value, dict) else {}
        tool = value.get('tool')
        return {'tool': tool if isinstance(tool, str) and tool in allowed_tools else '[REDACTED_TOOL]',
                'arguments_omitted': True}
    def program(value):
        if not isinstance(value, dict) or not isinstance(value.get('program'), list):
            return None
        return [call(step) for step in value['program']]
    attempts = []
    for attempt in trace.get('planner_attempts', []):
        raw = attempt.get('raw_response')
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            parsed = None
        attempts.append({'attempt': number(attempt.get('attempt')), 'status': status(attempt.get('status')),
                         'tools': program(parsed), 'unparseable': program(parsed) is None})
    programs = []
    for item in trace.get('programs', []):
        programs.append({'program_index': number(item.get('program_index')), 'tools': program(item.get('plan')),
            'status': status(item.get('status')), 'terminated': item.get('terminated') is True,
            'failure_index': number(item.get('failure_index')),
            'steps': [{'call': call(step.get('call')), 'status': status(step.get('status'))}
                      for step in item.get('steps', [])]})
    final = trace.get('final_result')
    projected_final = None
    if isinstance(final, dict):
        category, grade = final.get('category'), final.get('grading_level')
        projected_final = {'category_alias': category_aliases.get(category) if isinstance(category,str) else None,
                           'grade_alias': grade_aliases.get(grade) if isinstance(grade,str) else None}
    stop = trace.get('stop_reason')
    budget = trace.get('budget', {})
    return {'public_id': public_id, 'source_trace_sha256': trace_sha256,
        'privacy': {'lossy': True, 'arguments_and_state_references_omitted': True,
                    'category_and_grade_are_opaque_aliases': True, 'not_training_data': True,
                    'full_replay_supported': False, 'gold_and_private_source_ids_included': False},
        'receipt_status': status(receipt_status), 'terminated': trace.get('terminated') is True,
        'stop_reason': stop if isinstance(stop,str) and stop in stops else 'other',
        'planner_attempts': attempts, 'programs': programs, 'final_result': projected_final,
        'budget': {key: number(budget.get(key)) for key in ('programs','planner_attempts','skill_steps') if key in budget}}
