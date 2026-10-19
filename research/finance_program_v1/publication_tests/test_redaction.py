import copy
import importlib.util
import json
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

class RedactionTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('redact_trajectories'), 'public trajectory redactor must exist')
        from redact_trajectories import public_trace
        return public_trace

    def trace(self):
        plan={'program':[{'tool':'GET_FIELD_METADATA','arguments':{}},{'tool':'FINALIZE_CLASSIFICATION','arguments':{'final_evidence_state':'$state.final_evidence_state'}}]}
        return {'input':{'field_name':'PRIVATE_FIELD','source_id':'secret-source'},'terminated':True,'stop_reason':'validated_terminal',
            'planner_attempts':[{'attempt':1,'status':'accepted','messages':[{'role':'user','content':'PRIVATE_FIELD secret-source'}],'raw_response':json.dumps(plan)}],
            'programs':[{'program_index':1,'plan':plan,'status':'terminated','terminated':True,'failure_index':None,'steps':[
                {'step_index':0,'call':plan['program'][0],'status':'success','summary':'PRIVATE_FIELD secret-source', 'data':{'field_metadata':{'password':'DO_NOT_PUBLISH','host':'192.0.2.1'}}}]}],
            'final_result':{'category':'finance:synthetic/category','grading_level':'L1','private':'secret-source'},'budget':{'programs':1}}

    def test_actual_program_and_outcome_preserved_without_private_text(self):
        fn=self.api(); t=self.trace(); original=copy.deepcopy(t); result=fn(t,'teacher-0001','a'*64)
        self.assertEqual(result['programs'][0]['plan'],t['programs'][0]['plan'])
        self.assertEqual(result['final_result'],{'category':'finance:synthetic/category','grading_level':'L1'})
        self.assertTrue(result['terminated']);self.assertEqual(t,original)
        text=json.dumps(result)
        for private in ['PRIVATE_FIELD','secret-source','DO_NOT_PUBLISH','192.0.2.1']:
            self.assertNotIn(private,text)
        self.assertFalse(result['privacy']['full_controller_replay_supported'])

    def test_failures_and_invalid_planner_attempts_not_deleted(self):
        fn=self.api();t=self.trace();t['terminated']=False;t['final_result']=None;t['stop_reason']='planner_budget_exhausted'
        t['planner_attempts'].append({'attempt':2,'status':'rejected','raw_response':'INVALID PRIVATE_FIELD secret-source','error':'secret-source'})
        result=fn(t,'student-0001','b'*64)
        self.assertFalse(result['terminated']);self.assertEqual(len(result['planner_attempts']),2)
        self.assertTrue(result['planner_attempts'][1]['raw_response_unparseable'])
        self.assertNotIn('secret-source',json.dumps(result))

    def test_literal_dynamic_arguments_redacted_and_disclosed(self):
        fn=self.api();t=self.trace();t['programs'][0]['plan']['program'][0]['arguments']={'field_name':'PRIVATE_FIELD'}
        result=fn(t,'sample-0001','c'*64)
        self.assertEqual(result['programs'][0]['plan']['program'][0]['arguments']['field_name'],'[REDACTED]')
        self.assertTrue(result['privacy']['literal_arguments_redacted'])

    def test_trace_identity_required(self):
        fn=self.api()
        with self.assertRaises(ValueError):fn(self.trace(),'sample-0001','bad')

if __name__=='__main__': unittest.main()
