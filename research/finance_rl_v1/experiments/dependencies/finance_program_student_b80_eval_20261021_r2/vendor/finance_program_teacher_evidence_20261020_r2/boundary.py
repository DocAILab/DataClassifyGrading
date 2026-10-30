"""Persistent per-trajectory negative evidence; immutable v1 validation is reused."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('gap_original_boundary',
    ROOT.parent / 'finance_program_teacher_evidence_20261020_r1/boundary.py')
v1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v1)

raw_json, digest, strict = v1.raw_json, v1.digest, v1.strict
evidence_view, complete_optional = v1.evidence_view, v1.complete_optional
classification_response, skill_name = v1.classification_response, v1.skill_name
CLASSIFIERS, IDENTIFIERS, BUSINESS_LEAVES = v1.CLASSIFIERS, v1.IDENTIFIERS, v1.BUSINESS_LEAVES
STOP_REASON = 'unresolved_no_new_business_evidence'
GAP_POLICY = ('Persistent evidence gaps: evidence_gap_ledger records previous structurally valid insufficient outcomes '
    'and their acquired-business-fact checkpoints. Retain these unresolved distinctions in planning and Skill decisions. '
    'A new or value-changed eligible source attribute only permits reconsideration; a repeated acquisition path '
    'for the same attribute is not new evidence. This does not certify relevance, truth, '
    'or sufficiency. Identifiers, provenance, availability updates, hypotheses, and candidate descriptions do not resolve gaps. '
    'Acquire relevant available metadata before another classification. Repeating classification with no new eligible '
    'raw fact after an insufficient outcome is locally stopped as unresolved.')


def checkpoint(view):
    """Checkpoint source attributes, retaining literal fact IDs in the evidence view."""
    return {source_identity(view, fid): deepcopy(fact['value']) for fid, fact in view['facts'].items()
            if fact.get('decisive_eligible') is True and fact.get('role') == 'raw_fact'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def source_identity(view, fid):
    """Normalize only aliases produced by the native, privately scoped lookup.

    All neighbor projections use the same deterministic first row per field name
    in this trajectory's single actual table. No text-only global deduplication.
    """
    if fid.startswith('metadata.'):
        return fid
    parts = fid.split('.')
    wrapper = '.'.join(parts[:2])
    obj = view['facts'].get(wrapper, {}).get('value')
    if parts[:3] == ['observations', 'table_context', 'target']:
        return 'metadata.' + '.'.join(parts[3:])
    parents = []
    for part in parts[2:]:
        parents.append(obj)
        if isinstance(obj, dict): obj = obj.get(part)
        elif isinstance(obj, list) and part.isdigit(): obj = obj[int(part)]
        else: return fid
    attribute = parts[-1]
    for parent in reversed(parents):
        if isinstance(parent, dict) and 'field_name' in parent:
            return 'same_table_field.' + canonical([parent['field_name'], attribute])
    if wrapper == 'observations.database_context':
        if attribute == 'database_description': return 'metadata.database_description'
        if 'table_descriptions' in parts:
            table = next((p['table_name'] for p in parents if isinstance(p, dict) and 'table_name' in p), None)
            target = view['facts'].get('metadata.table_name', {}).get('value')
            if table == target and target is not None: return 'metadata.table_description'
            return 'table_description.' + canonical(table)
    return fid


def eligible_gains(view, previous):
    """Return citation/audit IDs whose actual source identity and value are new."""
    return [fid for fid, fact in view['facts'].items()
            if fact.get('decisive_eligible') is True and fact.get('role') == 'raw_fact'
            and (source_identity(view, fid) not in previous
                 or canonical(fact['value']) != canonical(previous[source_identity(view, fid)]))]


class Guard(v1.Guard):
    def __init__(self, generate):
        self.backend = generate
        self.gap_ledger = []
        self.stop_reason = None
        self._request_view = None
        super().__init__(self._dispatch)

    @property
    def halted(self):
        return getattr(self.backend, 'halted', False)

    @property
    def last_generation_metadata(self):
        if self.audits and self.audits[-1].get('backend_called') is False:
            return {}
        return getattr(self.backend, 'last_generation_metadata', {})

    def _dispatch(self, actual):
        # v1 constructs the acquired-only view and appends its audit before this
        # callback. Enrich that exact request, then bind hashes to what is sent.
        payload = json.loads(actual[-1]['content'])
        self._request_view = deepcopy(payload['evidence_boundary'])
        payload['evidence_gap_ledger'] = deepcopy(self.gap_ledger)
        actual[0]['content'] += '\n' + GAP_POLICY
        actual[-1]['content'] = raw_json(payload)
        audit = self.audits[-1]
        audit.update(actual_messages_sha256=digest(actual),
                     gap_ledger_sha256=digest(self.gap_ledger), backend_called=False)
        if audit['classification'] and self.gap_ledger:
            previous = self.gap_ledger[-1]['checkpoint']
            gained = eligible_gains(self._request_view, previous)
            audit['eligible_gain_fact_ids'] = gained
            if not gained:
                self.stop_reason = STOP_REASON
                audit['blocked_messages'] = deepcopy(actual)
                raise ValueError(STOP_REASON)
        audit['backend_called'] = True
        return self.backend(actual)

    def __call__(self, messages):
        try:
            return super().__call__(messages)
        except ValueError:
            audit = self.audits[-1] if self.audits else {}
            if audit.get('blocked_messages') is not None:
                audit.update(status='locally_blocked', error_kind=STOP_REASON, reason=STOP_REASON,
                             backend_called=False)
            elif audit.get('error_kind') == 'insufficient':
                self.gap_ledger.append(dict(
                    gap_index=len(self.gap_ledger) + 1,
                    audit_index=len(self.audits) - 1,
                    skill=audit['skill'],
                    missing_evidence=audit['support']['missing_evidence'],
                    checkpoint=checkpoint(self._request_view),
                    view_sha256=audit['view_sha256'],
                    provider_response_sha256=audit['provider_response_sha256']))
                audit['gap_index'] = len(self.gap_ledger)
            raise
