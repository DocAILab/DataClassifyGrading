"""Audit recorded failures without either hiding them or demanding all predictions."""


def generation_status(record,actual):
    if not actual['generation_invoked'] or not actual['generation_returned']:
        raise ValueError('missing actual generation execution')
    if actual['error'] is not None:
        if not record.get('error'):
            raise ValueError('actual generation error missing from native replay record')
        return 'recorded_generation_error'
    if record.get('error'):
        # The adapter may reject a returned format after successful EOS; the
        # native replay must still keep that error, separate from generation.
        if actual.get('termination_reason')!='eos' or (actual.get('output_token_ids') or [None])[-1] not in (248044,248046):
            raise ValueError('adapter error cannot hide missing EOS')
        return 'recorded_adapter_error'
    if actual.get('termination_reason')!='eos' or (actual.get('output_token_ids') or [None])[-1] not in (248044,248046):
        raise ValueError('successful generation lacks EOS')
    return 'eos_success'


def outcome_status(trace,row,generation_errors):
    if row['prediction']!=trace['final_result'] or row['stop_reason']!=trace['stop_reason']:
        raise ValueError('result differs from actual trace')
    if bool(row['algorithm_completed'])!=bool(trace['terminated']):
        raise ValueError('algorithm completion differs from trace')
    if trace['final_result'] is not None:
        if not trace['terminated'] or not row['legal_completion']:
            raise ValueError('prediction lacks legal terminal outcome')
        return 'guard_checked_prediction'
    if row['legal_completion']:
        raise ValueError('absent prediction cannot count as legal classification')
    if trace['terminated']:
        if trace['stop_reason']!='abstained_terminal':
            raise ValueError('unknown no-prediction terminal contract')
        return 'abstained_terminal'
    if generation_errors<1:
        raise ValueError('nonterminal output needs independently recorded generation failure')
    return 'generation_failure_preserved'
