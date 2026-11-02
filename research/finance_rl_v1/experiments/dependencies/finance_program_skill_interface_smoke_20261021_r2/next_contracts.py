"""Typed JSON prompts and observed-only recovery hints; no output repair."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
V1 = ROOT.parent / 'finance_program_skill_interface_smoke_20261021_r1'
spec = importlib.util.spec_from_file_location('previous_prompt_contracts', V1 / 'contracts.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)

RECOVERY_RULES = (
    'Recovery interface v2: use the observed recovery_interface, not the historical unexecuted tail as a required recipe. '
    'Do not repeat GET_FIELD_METADATA when field_metadata_already_queried=true: its actual result remains in state. '
    'Do not call an unbound interpretation; an omitted field_type/value is unknown, not an instruction to invent it. '
    'If that attribute is decision-critical, acquire that missing metadata type once using the existing advertised contract; '
    'otherwise skip the optional interpretation and reuse valid evidence. Never treat an unknown field type as a mandatory blocker for all classification. '
    'If classification reported insufficient, do not repeat classification until an actually new decision-relevant raw fact is acquired. '
    'Select a relevant still-unqueried bound acquisition from the original library; the listed context options are choices, not automatic steps, not verified support, and may reveal no useful evidence. '
    'Do not repeat successful interpretation or selection with identical inputs. A changed raw fact can justify a new interpretation, but repetition alone cannot. '
    'For a schema/type failure, repair the requested Skill format rather than repeating successful acquisition. '
    'If no available action can resolve the distinction, preserve insufficient; do not force a category, grade or finalization. '
    'Return only the original Program JSON, obey original dependencies and remaining budget, and do not add reason/stop/new tools.')


def expand(value, shared, trail=()):
    if isinstance(value, dict):
        if set(value) == {'prompt_ref'}:
            key = value['prompt_ref']
            if key in trail:
                raise ValueError('cyclic original PromptRef')
            return expand(shared[key], shared, trail + (key,))
        return {k: expand(v, shared, trail) for k, v in value.items()}
    if isinstance(value, list):
        return [expand(v, shared, trail) for v in value]
    return value


def recovery_interface(payload):
    feedback = expand(payload.get('feedback'), payload.get('shared_values', {}))
    if not feedback:
        return None
    view = payload.get('evidence_boundary', {})
    facts = view.get('facts', {})
    metadata_keys = {k.removeprefix('metadata.') for k in facts if k.startswith('metadata.')}
    metadata_keys.update(payload.get('current', {}).get('input', {}))
    ledger = view.get('availability_ledger', [])
    attempted_skills = {v.get('skill') for v in ledger if v.get('status') in ('obtained', 'unavailable')}
    possible = []
    observation_for = {'GET_TABLE_FIELDS': 'observations.table_fields', 'GET_COMPLETE_TABLE_CONTEXT': 'observations.table_context', 'GET_DATABASE_CONTEXT': 'observations.database_context'}
    for skill in payload.get('skill_library', []):
        name = skill['name']
        if name not in observation_for or name in attempted_skills:
            continue
        if observation_for[name] in facts:
            continue
        if all(key in metadata_keys for key in skill.get('required_inputs', [])):
            possible.append(name)
    failures = []
    if isinstance(feedback, dict):
        for step in feedback.get('steps', []):
            if step.get('status') != 'success':
                failures.append(dict(tool=step.get('call', {}).get('tool'), status=step.get('status'),
                                     summary=step.get('summary', '')[:600]))
        if feedback.get('error'):
            failures.append(dict(tool='Planner', status=feedback.get('status'), summary=str(feedback['error'])[:600]))
    advertised = {v['name'] for v in payload.get('skill_library', [])}
    observed_tools = {v['tool'] for v in failures}
    return dict(schema='observed-only-recovery-interface-v2', last_failures=failures,
        field_metadata_already_queried=any(v.get('scope') == 'query' and v.get('type') == 'field_metadata'
            and v.get('status') in ('obtained', 'unavailable') for v in ledger),
        unbound_interpretations=[name for name, key in [('INTERPRET_FIELD_TYPE', 'field_type'), ('INTERPRET_FIELD_VALUE', 'value')]
            if key not in metadata_keys and name in advertised | observed_tools],
        missing_metadata=deepcopy(view.get('missing_metadata', [])),
        unqueried_context_options=possible,
        unresolved_gaps=[dict(missing_evidence=v.get('missing_evidence', '')) for v in payload.get('evidence_gap_ledger', [])],
        authority='Only a summary of already visible observations. No newly acquired facts, readiness guarantee or category target.')


def rewrite(messages, condition):
    if condition not in ('original', 'unified'):
        raise ValueError('unknown condition')
    result = previous.rewrite(messages, 'unified')
    if condition == 'original':
        return result
    system = result[0]['content']
    match = re.match(r'Execute only the selected Skill ([A-Z_]+)\.', system)
    name = match.group(1) if match else None
    if name in previous.CLASSIFIERS:
        result[0]['content'] += (
            '\nTyped value reminder: support.missing_evidence MUST be a JSON string, never null, never an object. '
            'If support.sufficient is true, use exactly "missing_evidence":"". '
            'If support.sufficient is false, use a nonempty missing_evidence string, empty citations, and option_id="UNRESOLVED". '
            'The presence of an acquired quote does not prove the category follows; keep the original sufficiency/evidence rules. '
            'All three root keys and all three support keys remain mandatory. Do not emit a schema or fragment.')
    elif name == 'SYNTHESIZE_METADATA_SEMANTICS':
        result[0]['content'] += (
            '\nSynthesis output interface v2: Output exactly one valid JSON object with only summary and data. '
            'summary MUST be a nonempty JSON string. data has exactly one key metadata_semantics. '
            'metadata_semantics MUST be one nonempty JSON string, not an object or array. '
            'Describe direct observed evidence, auxiliary model hypotheses, and unresolved uncertainty inside that ONE string in natural language. '
            'Do NOT create strong_evidence/auxiliary_evidence/uncertain_hypotheses JSON keys or arrays. '
            'No Markdown fences or trailing text. Use only original supplied evidence; no category/grade, invented metadata or automatic consistency claims. '
            'Output shape: {"summary":"concise synthesis","data":{"metadata_semantics":"Direct observations: ...; auxiliary hypotheses: ...; unresolved: ..."}}. '
            'Replace example text with actual supplied evidence; do not copy ellipses.')
    elif name is None:
        payload = json.loads(result[-1]['content'])
        hint = recovery_interface(payload)
        if hint is not None:
            payload['recovery_interface'] = hint
            result[0]['content'] += '\n' + RECOVERY_RULES
            result[-1]['content'] = json.dumps(payload, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return result


def validate_output(stage, messages, raw, grounding, boundary):
    if stage != 'SYNTHESIZE_METADATA_SEMANTICS':
        return previous.validate_output(stage, messages, raw, grounding, boundary)
    result = dict(json_valid=False, schema_valid=False, protocol_valid=False, accepted_success=False, insufficient=False, error=None)
    try:
        value = boundary.strict(raw)
        result['json_valid'] = isinstance(value, dict)
        if not isinstance(value, dict) or set(value) != {'summary', 'data'} or not isinstance(value['summary'], str) or not value['summary'].strip():
            raise ValueError('exact native summary/data required')
        data = value['data']
        if not isinstance(data, dict) or set(data) != {'metadata_semantics'} or not isinstance(data['metadata_semantics'], str) or not data['metadata_semantics'].strip():
            raise ValueError('metadata_semantics violates its declared evidence type')
        result.update(schema_valid=True, protocol_valid=True, accepted_success=True)
    except (ValueError, TypeError, KeyError) as exc:
        result['error'] = str(exc)
    return result
