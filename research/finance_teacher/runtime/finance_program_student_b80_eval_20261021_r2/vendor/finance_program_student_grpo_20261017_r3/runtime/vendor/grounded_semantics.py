"""Versioned extractive grounding, not an automatic semantic-truth judge."""
from copy import deepcopy
import json

TARGETS = {'EXPAND_FIELD_ABBREVIATION': 'field_abbreviation_semantics',
           'INTERPRET_FIELD_NAME': 'field_semantics',
           'INTERPRET_FIELD_DESCRIPTION': 'field_description_semantics'}


def validate_value(value, context):
    keys = {'schema', 'status', 'field_name', 'description', 'source_consistency'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('grounded evidence requires exact typed fields; free interpretation prohibited')
    if value['schema'] != 'grounded-source-v1' or value['source_consistency'] != 'not_adjudicated':
        raise ValueError('grounding schema or non-adjudication status differs')
    if not isinstance(context.get('field_name'), str) or not context['field_name'].strip() or value['field_name'] != context['field_name']:
        raise ValueError('grounded field identity differs')
    desc = context.get('field_description')
    observed = isinstance(desc, str) and bool(desc.strip()) and desc.strip().casefold() not in ('none', 'null', 'n/a', 'nan')
    if observed:
        if value['status'] != 'observed_description' or value['description'] != desc:
            raise ValueError('grounded description must equal entire actual acquired description')
    elif value['status'] != 'unknown' or value['description'] is not None:
        raise ValueError('absent acquired description permits unknown only')
    return True


def valid_shape(value):
    try:
        if not isinstance(value, dict):return False
        validate_value(value, {'field_name': value.get('field_name'), 'field_description': value.get('description')})
    except (ValueError, TypeError):return False
    return True


def make_messages(contract, resolved, context):
    assert contract.name in TARGETS
    system = (f'Execute only the selected Skill {contract.name}.\n'
        f'Versioned contract:\n{contract.body}\n'
        'Use only resolved_inputs and the explicitly declared grounding_context from already acquired raw metadata. '
        'Return exactly summary and data. summary must equal the status string. '
        'The produced value must be an object with exactly schema,status,field_name,description,source_consistency. '
        'schema="grounded-source-v1"; field_name must copy grounding_context.field_name; '
        'source_consistency="not_adjudicated" always. If grounding_context contains a real field_description, '
        'status="observed_description" and description must copy that entire original string exactly. '
        'Otherwise status="unknown" and description=null. No guessed expansion, translation, paraphrase, '
        'candidate meanings, extra keys or assertion that name and description agree/conflict. '
        'Observed description is a source observation, not semantic truth or verified consistency. '
        'Do not retrieve hidden metadata, labels, registry outcomes or future observations.')
    payload = {'skill': contract.name, 'resolved_inputs': deepcopy(resolved), 'grounding_context': deepcopy(context),
               'grounding_protocol': 'extractive-grounded-source-v1',
               'output_schema': {'summary': 'same as status', 'data': {TARGETS[contract.name]: {
                   'schema': 'grounded-source-v1', 'status': 'observed_description or unknown',
                   'field_name': 'copy exact context field_name', 'description': 'exact context description or null',
                   'source_consistency': 'not_adjudicated'}}}}
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]


def validate_observation(observation, context):
    key = TARGETS[observation.skill]
    value = observation.data[key]
    validate_value(value, context)
    if observation.summary != value['status']:raise ValueError('summary must equal the grounded status, not free prose')
    return observation
