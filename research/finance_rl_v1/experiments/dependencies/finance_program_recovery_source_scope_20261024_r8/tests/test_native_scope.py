"""One actual deterministic GET path, no model and no category shortcut."""
import importlib.util,json,sys,unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PARENT=ROOT.parent/'finance_program_student_fresh162_20261022_r5'
sys.path.append(str(PARENT))
spec=importlib.util.spec_from_file_location('source_scope_native_parent_runtime',PARENT/'eval_runtime.py')
parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
engine=parent.load_engine(parent.E2)
class NativeScopeTests(unittest.TestCase):
    def test_native_metadata_get_then_table_get_preserves_guards_and_true_missing_db(self):
        inputs=json.loads((PARENT/'eval-inputs.json').read_bytes());study=inputs['study']
        scopes=json.loads((ROOT/'source-scopes.json').read_bytes())['rows']
        original=engine.old.c.controller
        try:
            from runtime import scoped_controller
            factory=scoped_controller(original,scopes)
        except ImportError:factory=original
        ctrl=factory(lambda _:self.fail('no model call permitted'),study)
        from agent.orchestration.contracts import MetadataRecord,ToolCall
        source_id='04f5a504-a17f-55cf-a2ba-8eb76898d522'
        record=MetadataRecord.from_mapping(dict(domain='finance',field_name='RESFULL_NM',source_id=source_id))
        state={};ctrl.bridge.availability.initialize(record,state)
        self.assertNotIn('field_metadata',state)
        counts=Counter();lineage=dict(candidates=0,evidence=0,selection=0)
        first=ctrl.bridge.step(ToolCall('GET_FIELD_METADATA',{}),record,state,counts,lineage)
        self.assertEqual(first['status'],'success');self.assertNotIn('database_name',state['current_metadata'])
        second=ctrl.bridge.step(ToolCall('GET_TABLE_FIELDS',{}),record,state,counts,lineage)
        self.assertEqual(second['status'],'success')
        self.assertGreater(second['data']['table_fields']['returned_count'],0)
        self.assertIsNone(second['data']['table_fields']['database_name'])
        self.assertEqual(second['data']['table_fields']['missing_locators'],['database_name'])
        self.assertIs(engine.old.c.controller,original)
        # No change to native typed bindings, citation or terminal functions.
        from pool_runtime import _binding
        self.assertIsNone(_binding('table_fields',record,state))
        view=engine.boundary.evidence_view(record,state)
        actual=[f['value'] for k,f in view['facts'].items() if k.startswith('observations.table_fields.fields.') and f['decisive_eligible']]
        self.assertIn('功能路径(action)',actual)
        self.assertFalse(ctrl.bridge.guard._valid_terminal(dict(category='invented',grading_level='L1'),state))
if __name__=='__main__':unittest.main()
