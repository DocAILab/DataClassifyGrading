"""Prompt-only serialization alignment; never repairs or relaxes model outputs."""
from copy import deepcopy
import json
import re

GROUNDED = {'INTERPRET_FIELD_NAME': 'field_semantics',
            'INTERPRET_FIELD_DESCRIPTION': 'field_description_semantics',
            'EXPAND_FIELD_ABBREVIATION': 'field_abbreviation_semantics'}
CLASSIFIERS = {'INFER_CATEGORY', 'INFER_CATEGORY_FROM_TOPK'}
JSON_ONLY = ('Output exactly one valid JSON object, beginning with { and ending with }. '
             'No YAML, Markdown fences, XML, preamble, explanation or trailing text. '
             'Use double-quoted JSON keys/strings, JSON null and lowercase true/false. ')


def rewrite(messages, condition):
    if condition not in ('original', 'unified'):
        raise ValueError('unknown condition')
    result = deepcopy(messages)
    if condition == 'original':
        return result
    system = result[0]['content']
    match = re.match(r'Execute only the selected Skill ([A-Z_]+)\.', system)
    name = match.group(1) if match else None
    if name in GROUNDED:
        key = GROUNDED[name]
        schema = dict(type='object', additionalProperties=False, required=['summary', 'data'], properties={
            'summary': dict(type='string', enum=['observed_description', 'unknown']),
            'data': dict(type='object', additionalProperties=False, required=[key], properties={key: dict(
                type='object', additionalProperties=False,
                required=['schema', 'status', 'field_name', 'description', 'source_consistency'],
                properties={'schema': {'const': 'grounded-source-v1'},
                            'status': {'enum': ['observed_description', 'unknown']},
                            'field_name': {'type': 'string'}, 'description': {'type': ['string', 'null']},
                            'source_consistency': {'const': 'not_adjudicated'}})})})
        head, sep, tail = system.partition('\n')
        system = head + '\n' + JSON_ONLY + '\n' + tail
        system += '\nJSON serialization contract (the source-grounding rules above remain unchanged): ' + json.dumps(schema, separators=(',', ':'))
        system += '\nReturn the JSON instance only, not this schema. Copy only actual grounding_context values; summary equals status.'
    elif name in CLASSIFIERS:
        pattern = r'Return exactly one JSON object:.*?(?=\nEvidence boundary v1:)'
        system, count = re.subn(pattern,
            'Do not choose outside candidate_details. Keep summary concise (at most 20 characters).',
            system, count=1, flags=re.S)
        if count != 1:
            raise ValueError('unknown classifier output prompt; refusing broad replacement')
        start = 'Versioned output extension: return exactly option_id, summary, support. '
        legacy = 'support={sufficient:boolean,citations:[{fact_id:string,quote:string}],missing_evidence:string}. '
        if system.count(start + legacy) != 1:
            raise ValueError('unknown support extension; refusing ambiguous rewrite')
        schema = dict(type='object', additionalProperties=False, required=['option_id', 'summary', 'support'], properties={
            'option_id': dict(type='string'), 'summary': dict(type='string'),
            'support': dict(type='object', additionalProperties=False,
                required=['sufficient', 'citations', 'missing_evidence'], properties={
                    'sufficient': dict(type='boolean'), 'missing_evidence': dict(type='string'),
                    'citations': dict(type='array', maxItems=8, items=dict(type='object', additionalProperties=False,
                        required=['fact_id', 'quote'], properties={'fact_id': {'type': 'string'}, 'quote': {'type': 'string'}}))})})
        system = system.replace(start + legacy, 'The sole output contract is: ' + JSON_ONLY +
            json.dumps(schema, separators=(',', ':')) + '\nReturn the JSON instance, not the schema. ')
        head, sep, tail = system.partition('\n')
        system = head + '\n' + JSON_ONLY + '\n' + tail
    result[0]['content'] = system
    return result


def validate_output(stage, messages, raw, grounding, boundary):
    result = dict(json_valid=False, schema_valid=False, protocol_valid=False,
                  accepted_success=False, insufficient=False, error=None)
    try:
        value = boundary.strict(raw)
        if not isinstance(value, dict):
            raise ValueError('one JSON object required')
        result['json_valid'] = True
        payload = json.loads(messages[-1]['content'])
        if stage in GROUNDED:
            key = GROUNDED[stage]
            if (set(value) != {'summary', 'data'} or not isinstance(value['data'], dict)
                    or set(value['data']) != {key}):
                raise ValueError('exact native observation keys required')
            grounding.validate_value(value['data'][key], payload['grounding_context'])
            if value['summary'] != value['data'][key]['status']:
                raise ValueError('summary must equal grounded status')
            result.update(schema_valid=True, protocol_valid=True, accepted_success=True)
        else:
            try:
                boundary.classification_response(raw, payload['evidence_boundary'])
            except ValueError as exc:
                if str(exc).startswith('insufficient:'):
                    if value['option_id'] != 'UNRESOLVED':
                        raise ValueError('insufficient option must be UNRESOLVED')
                    result.update(schema_valid=True, protocol_valid=True, insufficient=True)
                    return result
                raise
            choices = {c['option_id'] for c in payload.get('candidate_details', [])}
            if choices and value['option_id'] not in choices:
                raise ValueError('option outside actual candidate_details')
            result.update(schema_valid=True, protocol_valid=True, accepted_success=True)
    except (ValueError, TypeError, KeyError) as exc:
        result['error'] = str(exc)
    return result
