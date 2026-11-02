"""Summarize actual outer-loop observations without proposing or executing steps."""
from .contracts import json_snapshot


def recovery_context(state, budget, feedback):
    """Return detached JSON facts; presence and historical success imply no readiness.

    Feedback is the controller's last observation, not model-authored recovery advice.
    Keep raw feedback at its existing boundary; this summary omits values and arguments.
    """
    previous = None
    if feedback and feedback.get('status') in ('completed', 'failed', 'budget_exhausted', 'terminated') and 'plan' in feedback:
        steps = feedback.get('steps', [])
        executed = {step['step_index'] for step in steps}
        previous = {
            'status': feedback['status'],
            'failure_index': feedback.get('failure_index'),
            'successful_steps': [{'step_index': s['step_index'], 'tool': s['call']['tool']}
                                 for s in steps if s['status'] == 'success'],
            'failed_steps': [{'step_index': s['step_index'], 'tool': s['call']['tool'], 'status': s['status']}
                             for s in steps if s['status'] != 'success'],
            'unexecuted_tail': [{'step_index': i, 'tool': call['tool']}
                                for i, call in enumerate(feedback['plan']['program']) if i not in executed],
        }
    return json_snapshot({
        'previous_program': previous,
        'classification_completed': bool(previous and previous['status'] == 'terminated'
                                         and feedback.get('terminated') is True
                                         and feedback.get('steps')
                                         and feedback['steps'][-1]['status'] == 'success'),
        'remaining_budget': budget,
        'comparison_failures': state.get('comparison_failure_ledger', []),
        'invalidations': feedback.get('invalidations', []) if previous else [],
        'observed_state_keys': sorted(state),
        'state_rule': 'Observed keys indicate presence only; neither presence nor historical success verifies readiness. Removed invalidated outputs are not available in current state.',
        'availability_observations': [{k: entry[k] for k in ('scope', 'type', 'status', 'skill') if k in entry}
                                      for entry in state.get('availability_ledger', [])],
        'availability_rule': 'These are recorded observations, not new checks. Missing observations remain unknown/unqueried; omitted batch attributes remain unknown. Obtained and canonical unavailable are distinct; error is retryable and does not establish absence. Target details remain in actual public state.',
        'tail_rule': 'Unexecuted tail is historical intent only, without arguments; choose a new whole Program autonomously using current state and contracts.',
        'task_reminder': 'Consider the remaining complete classification task within the actual global budget. Only a validated successful terminal completes classification. A successful recovery step alone may leave the task unfinished, especially on the last available Program. Partial legal Programs remain allowed; no steps or extra budget are supplied.',
    })
