import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from redact_trajectories import public_trace


class PublicationTests(unittest.TestCase):
    def project(self, trace):
        return public_trace(trace,'val-0001','a'*64,'completed',{'GET_FIELD_METADATA'},
                            {'private-standard':'standard-001'},{'private-grade':'grade-001'})

    def test_no_free_text_or_dynamic_keys_escape(self):
        trace={'input':{'id':'PRIVATE_SOURCE','field_name':'PRIVATE_FIELD'},
          'terminated':True,'stop_reason':'validated_terminal',
          'planner_attempts':[{'attempt':1,'status':'accepted','raw_response':json.dumps({'program':[
            {'tool':'GET_FIELD_METADATA','arguments':{'PRIVATE_FIELD':'$state.PRIVATE_FIELD'}},
            {'tool':'PRIVATE_FIELD','arguments':{}}]})}],
          'programs':[{'program_index':1,'status':'terminated','terminated':True,'plan':{'program':[]},
            'steps':[{'call':{'tool':'GET_FIELD_METADATA','arguments':{'x':'PRIVATE_SOURCE'}},
              'status':'success','summary':'PRIVATE_FIELD','data':{'PRIVATE_FIELD':'SECRET_VALUE'}}]}],
          'final_result':{'category':'private-standard','grading_level':'private-grade','reason':'SECRET_VALUE'},
          'budget':{'programs':1,'PRIVATE_FIELD':'SECRET_VALUE'}}
        original=copy.deepcopy(trace)
        result=self.project(trace)
        self.assertEqual(trace,original)
        serialized=json.dumps(result)
        for token in ('PRIVATE_SOURCE','PRIVATE_FIELD','SECRET_VALUE','private-standard','private-grade'):
            self.assertNotIn(token,serialized)
        self.assertEqual(result['final_result']['category_alias'],'standard-001')
        self.assertTrue(result['terminated'])

    def test_rejected_attempt_and_failure_preserved_without_raw_text(self):
        result=self.project({'terminated':False,'stop_reason':'planner_budget_exhausted',
            'planner_attempts':[{'attempt':1,'status':'rejected','raw_response':'SECRET invalid JSON'}],
            'programs':[],'final_result':None})
        self.assertEqual(len(result['planner_attempts']),1)
        self.assertTrue(result['planner_attempts'][0]['unparseable'])
        self.assertNotIn('SECRET',json.dumps(result))
        self.assertFalse(result['terminated'])

    def test_unknown_status_stop_and_budget_strings_are_not_published(self):
        result=self.project({'stop_reason':'PRIVATE_REASON','budget':{'programs':'PRIVATE_BUDGET'},
            'planner_attempts':[{'status':'PRIVATE_STATUS','attempt':'PRIVATE_COUNTER'}]})
        self.assertNotIn('PRIVATE_',json.dumps(result))

    def test_released_rows_and_metrics_agree(self):
        rows=[json.loads(line) for line in (ROOT/'trajectories/val397.redacted.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual([r['public_id'] for r in rows],[f'val-{i:04d}' for i in range(1,398)])
        self.assertEqual(sum(r['terminated'] for r in rows),115)
        self.assertEqual(sum(r['receipt_status']=='backend_error' for r in rows),17)
        summary=json.loads((ROOT/'results/evidence-summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['metrics']['joint_correct'],26)
        self.assertEqual(summary['metrics']['denominator'],397)
        self.assertTrue(all(r['privacy']['not_training_data'] for r in rows))

    def test_publication_code_has_only_declared_differences(self):
        import hashlib
        manifest=json.loads((ROOT/'results/code-publication.json').read_text(encoding='utf-8'))
        changes=[name for name,digest in manifest['historical_runtime_files_sha256'].items()
                 if hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()!=digest]
        self.assertEqual(sorted(changes),sorted(manifest['publication_changes']))

if __name__=='__main__': unittest.main()
