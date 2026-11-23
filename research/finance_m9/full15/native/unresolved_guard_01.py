"""Isolated experimental wrapper; does not alter frozen native code."""
from copy import deepcopy


def finish(trace, controller, record, original_finish):
    backend_error = any(
        audit.get('status') == 'generation_error' and audit.get('error_kind') == 'backend_error'
        for audit in trace.get('evidence_audits', [])
    )
    if (trace.get('terminated') or backend_error
            or trace.get('stop_reason') != 'unresolved_no_new_business_evidence'):
        return original_finish(trace, controller, record)
    result = deepcopy(trace)
    result.update(terminated=True, final_result=None, stop_reason='abstained_terminal',
                  termination={'policy': 'unresolved-guard-experimental-01',
                               'outcome': 'abstained',
                               'reason': 'unresolved_no_new_business_evidence',
                               'original_stop_reason': trace['stop_reason'],
                               'evidence_sufficient': False, 'requires_review': True,
                               'new_model_calls': 0},
                  terminal_event={'tool': 'ABSTAIN_CLASSIFICATION', 'status': 'success', 'model_calls': 0})
    return result
