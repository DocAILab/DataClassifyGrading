"""Use actual archived classifier requests; no prediction/model replay."""
import copy
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent/'finance_program_recovery_source_scope_20261024_r9'
sys.path.insert(0,str(ROOT))

def load_previous():
    # R9 is an exact alias of this sealed R5 file; avoid rebinding its `common` alias.
    path=ROOT.parent/'finance_program_recovery_feedback_20261024_r5/interface.py'
    spec=importlib.util.spec_from_file_location('r9_delta_baseline_interface',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def session():
    target=ROOT/'interface.py'
    if not target.exists():return load_previous().RecoverySession()
    spec=importlib.util.spec_from_file_location('new_delta_interface',target)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.RecoverySession()

def classifier_messages(slot):
    result=[]
    fixture=BASE/'reviews/final-01/extract/runs'/slot
    if not fixture.exists():fixture=BASE/'runs'/slot
    for p in sorted(fixture.glob('call-*/returned.json')):
        raw=json.loads(p.read_bytes())['messages']
        messages=[dict(role=m['role'],content=m['content']) for m in raw]
        stage=re.match(r'Execute only the selected Skill ([A-Z_]+)\.',messages[0]['content'])
        if stage and stage.group(1) in ('INFER_CATEGORY_FROM_TOPK','INFER_CATEGORY'):result.append(messages)
    return result

def eligible(body):
    return {k:v for k,v in body['evidence_boundary']['facts'].items() if v.get('decisive_eligible') is True}

class EvidenceDeltaTests(unittest.TestCase):
    def test_real014_and018_new_context_indexed_against_previous_classification(self):
        for slot in ('014-N','018-N'):
            with self.subTest(slot=slot):
                tracker=session();requests=classifier_messages(slot);self.assertEqual(len(requests),2)
                first=json.loads(tracker.rewrite(requests[0])[-1]['content'])
                second=json.loads(tracker.rewrite(requests[1])[-1]['content'])
                self.assertIn('acquired_evidence_delta',second,'new raw observations need an explicit non-adjudicating delta')
                delta=second['acquired_evidence_delta'];before,after=eligible(first),eligible(second)
                self.assertEqual(delta['added_fact_ids'],sorted(after.keys()-before.keys()))
                self.assertGreater(len(delta['added_fact_ids']),20)
                self.assertFalse(delta['semantic_sufficiency_adjudicated'])
                self.assertFalse(delta['facts_added'])

    def test_same_facts_are_not_new_even_if_candidate_list_changes(self):
        tracker=session();messages=classifier_messages('014-N')[0]
        tracker.rewrite(messages);second=json.loads(tracker.rewrite(copy.deepcopy(messages))[-1]['content'])
        self.assertIn('acquired_evidence_delta',second)
        self.assertEqual(second['acquired_evidence_delta']['added_fact_ids'],[])
        self.assertEqual(second['acquired_evidence_delta']['changed_fact_ids'],[])

    def test_every_current_fact_and_other_payload_field_retained_exactly(self):
        old=load_previous().RecoverySession();new=session();seen=0
        for slot in ('003-N','012-N','014-N','015-N','016-N','018-N','030-N','037-N'):
            old=load_previous().RecoverySession();new=session()
            for messages in classifier_messages(slot):
                expected=json.loads(old.rewrite(messages)[-1]['content'])
                actual=json.loads(new.rewrite(messages)[-1]['content'])
                self.assertIn('acquired_evidence_delta',actual)
                delta=actual.pop('acquired_evidence_delta')
                self.assertEqual(actual,expected)
                self.assertTrue(set(delta['added_fact_ids'])<=set(expected['evidence_boundary']['facts']))
                self.assertNotIn('category',delta);self.assertNotIn('sufficient',delta)
                seen+=1
        self.assertEqual(seen,11,'exact archived classifier request count, not invented additional fixtures')

    def test_new_session_cannot_inherit_previous_case_fact_ids(self):
        first=session();first.rewrite(classifier_messages('014-N')[-1])
        second=json.loads(session().rewrite(classifier_messages('015-N')[0])[-1]['content'])
        self.assertIn('acquired_evidence_delta',second)
        self.assertFalse(second['acquired_evidence_delta']['has_previous_classification'])
        self.assertEqual(second['acquired_evidence_delta']['removed_fact_ids'],[])

    def test_initial_planner_bytes_remain_exact_baseline(self):
        p=BASE/'reviews/final-01/extract/runs/014-N/call-001/returned.json'
        if not p.exists():p=BASE/'runs/014-N/call-001/returned.json'
        raw=json.loads(p.read_bytes())['messages'];messages=[dict(role=m['role'],content=m['content']) for m in raw]
        self.assertEqual(session().rewrite(messages),load_previous().RecoverySession().rewrite(messages))

if __name__=='__main__':unittest.main()
