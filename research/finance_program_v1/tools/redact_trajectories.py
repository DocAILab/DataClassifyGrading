"""Public, lossy trajectory projection. Never use this output as training/replay data."""
import hashlib
import json
import re


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def public_trace(trace, public_id, trace_sha256):
    if not re.fullmatch(r'[0-9a-f]{64}', trace_sha256):
        raise ValueError('original trace SHA256 required')
    redacted = False

    def argument(value):
        nonlocal redacted
        if isinstance(value, str) and re.fullmatch(r'\$(?:state\.)?[A-Za-z_][A-Za-z0-9_.]*', value):
            return value
        redacted = True
        return '[REDACTED]'

    def call(value):
        tool = value.get('tool', '')
        if not isinstance(tool, str) or not re.fullmatch(r'[A-Z][A-Z0-9_]{0,100}', tool):
            tool = '[REDACTED_TOOL]'
        args = value.get('arguments', {})
        return {'tool': tool, 'arguments': {k: argument(v) for k, v in args.items()
            if re.fullmatch(r'[a-z][a-z0-9_]{0,80}', k)}}

    def plan(value):
        if not isinstance(value, dict) or not isinstance(value.get('program'), list):
            return None
        return {'program': [call(step) for step in value['program'] if isinstance(step, dict)]}

    attempts = []
    for a in trace.get('planner_attempts', []):
        raw = a.get('raw_response')
        try: parsed = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError): parsed = None
        projected = plan(parsed)
        attempts.append({'attempt': a.get('attempt'), 'status': a.get('status'),
            'program': projected, 'raw_response_unparseable': projected is None,
            'raw_response_sha256': digest(raw), 'error_present': bool(a.get('error'))})
    programs = []
    for p in trace.get('programs', []):
        steps = []
        for s in p.get('steps', []):
            data = s.get('data', {})
            steps.append({'step_index': s.get('step_index'), 'call': call(s.get('call', {})),
                'status': s.get('status'), 'observation_sha256': digest(s),
                'data_schema': {k: type(v).__name__ for k, v in data.items()
                    if re.fullmatch(r'[a-z][a-z0-9_]{0,80}', k)},
                'summary': '[REDACTED]', 'evidence_values_redacted': True})
        programs.append({'program_index': p.get('program_index'), 'plan': plan(p.get('plan')),
            'status': p.get('status'), 'terminated': p.get('terminated'),
            'failure_index': p.get('failure_index'), 'steps': steps})
    final = trace.get('final_result')
    result = {k: final[k] for k in ('category', 'grading_level') if k in final} if isinstance(final, dict) else None
    return {'public_id': public_id, 'source_trace_sha256': trace_sha256,
        'privacy': {'redacted': True, 'full_controller_replay_supported': False,
            'literal_arguments_redacted': redacted, 'not_training_data': True},
        'terminated': trace.get('terminated') is True, 'stop_reason': trace.get('stop_reason'),
        'planner_attempts': attempts, 'programs': programs, 'final_result': result,
        'budget': trace.get('budget', {})}
