"""Index exact current eligible source facts since the previous classifier call.

No copied fact values, relevance, preferred category or sufficiency verdict.
References point into the unchanged current native evidence boundary only.
"""
import hashlib
import json

def encoded(value):
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)

class EvidenceDelta:
    def __init__(self):
        self.previous=None

    def project(self,boundary):
        facts=boundary.get('facts',{})
        current={key:encoded(value) for key,value in facts.items()
                 if value.get('decisive_eligible') is True and value.get('role')=='raw_fact'}
        previous=self.previous
        before=previous or {}
        out=dict(schema='current-acquired-fact-delta-v1',
                 has_previous_classification=previous is not None,
                 added_fact_ids=sorted(current.keys()-before.keys()),
                 changed_fact_ids=sorted(key for key in current.keys() & before.keys() if current[key]!=before[key]),
                 removed_fact_ids=sorted(before.keys()-current.keys()),
                 unchanged_fact_count=sum(current[key]==before[key] for key in current.keys() & before.keys()),
                 current_eligible_fact_count=len(current),
                 current_eligible_view_sha256=hashlib.sha256(encoded(current).encode('utf-8')).hexdigest(),
                 semantic_sufficiency_adjudicated=False,facts_added=False,
                 meaning='IDs reference unchanged current evidence_boundary.facts. New relative to the previous classifier request is not proof of relevance, independence or sufficient support. Removed facts are not restored. Historical gaps are not automatically resolved.')
        self.previous=current
        return out
