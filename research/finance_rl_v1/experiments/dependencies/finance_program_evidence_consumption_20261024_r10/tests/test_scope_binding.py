"""Real037 acquired fixture through native binder and executor, no model/GPU."""
import copy
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent / 'finance_program_recovery_source_scope_20261024_r9'
PARENT = ROOT.parent / 'finance_program_student_fresh162_20261022_r5'
sys.path.insert(0, str(ROOT)); sys.path.append(str(PARENT))
spec = importlib.util.spec_from_file_location('scope_test_native_eval', PARENT / 'eval_runtime.py')
evaluation = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluation)
engine = evaluation.load_engine(evaluation.E2)
study = json.loads((PARENT / 'eval-inputs.json').read_bytes())['study']
_bootstrap_controller = engine.old.c.controller(lambda messages: '{}', study)
source_spec = importlib.util.spec_from_file_location('frozen_private_scope_provider', BASE / 'source_scope.py')
provider = importlib.util.module_from_spec(source_spec); source_spec.loader.exec_module(provider)
from agent.orchestration.contracts import MetadataRecord, ToolCall
from agent.orchestration.agent_loop import PolicyDecision
import pool_runtime

class ScopeBindingTests(unittest.TestCase):
    def setUp(self):
        fixture=BASE / 'reviews/final-01/extract/runs/037-N/trace.json'
        if not fixture.exists():fixture=BASE/'runs/037-N/trace.json'
        self.trace = json.loads(fixture.read_bytes())
        first = self.trace['initial']
        self.record = MetadataRecord(first['domain'], first['field_name'], {}, source_id=first['source_id'])
        self.state = copy.deepcopy(self.trace['state'])
        self.messages = []
        def generate(messages):
            self.messages.append(copy.deepcopy(messages))
            return json.dumps(dict(summary='CPU contract fixture only', data=dict(selected_context_fields=['URL'])))
        self.ctrl = engine.old.c.controller(generate, study)
        self.ctrl.bridge.executor.metadata_lookup = provider.SourceScopedMetadata(study['rows'], json.loads((BASE / 'source-scopes.json').read_bytes())['rows'])
        target = ROOT / 'scope_binding.py'
        if target.exists():
            spec = importlib.util.spec_from_file_location('candidate_scope_binding', target)
            candidate = importlib.util.module_from_spec(spec); spec.loader.exec_module(candidate)
            candidate.install(self.ctrl)

    def normalize(self):
        spec = self.ctrl.bridge.catalog['SELECT_CONTEXT_FIELDS']
        return self.ctrl.bridge.guard._normalize_policy_decision(
            PolicyDecision(ToolCall('SELECT_CONTEXT_FIELDS', {}), ''), self.record, self.state, spec)

    def test_original_rejects_real_acquired_scope_and_new_binding_accepts_exact(self):
        self.assertIsNone(pool_runtime._binding('table_fields', self.record, self.state))
        try:
            call = self.normalize().call
        except Exception as exc:
            self.fail('actual privately validated acquired scope must bind: ' + str(exc))
        self.assertEqual(call.arguments['table_fields'], '$state.table_fields')
        self.assertIsNone(self.state['table_fields']['database_name'])
        self.assertEqual(self.state['table_fields']['missing_locators'], ['database_name'])

    def test_real_selector_reaches_model_only_with_exact_acquired_payload(self):
        try:
            call = self.normalize().call
            observation = self.ctrl.bridge.executor.execute(call, self.record, self.state, self.ctrl.bridge.loader.load(call.tool))
        except Exception as exc:
            self.fail('native binder/executor boundary still blocks genuine acquired scope: ' + str(exc))
        self.assertEqual(observation.data['selected_context_fields'], ['URL'])
        self.assertEqual(len(self.messages), 1)
        actual = json.dumps(self.messages, ensure_ascii=False)
        for row in json.loads((BASE / 'source-scopes.json').read_bytes())['rows']:
            self.assertNotIn(row['source_table_id'], actual)

    def test_modified_acquired_field_rejected(self):
        self.state['table_fields']['fields'][0]['field_description'] = 'forged'
        with self.assertRaises(Exception): self.normalize()

    def test_forged_scope_marker_without_real_provider_rejected(self):
        self.ctrl.bridge.executor.metadata_lookup = pool_runtime.PoolMetadata(study['rows'])
        with self.assertRaises(Exception): self.normalize()

    def test_unacquired_state_rejected(self):
        self.state['_provenance'].pop('table_fields')
        with self.assertRaises(Exception): self.normalize()

    def test_wrong_producer_rejected(self):
        self.state['_provenance']['table_fields'] = 'INTERPRET_FIELD_NAME'
        with self.assertRaises(Exception): self.normalize()

    def test_empty_or_foreign_group_payload_rejected(self):
        for field in ('fields', 'table_name'):
            with self.subTest(field=field):
                original = copy.deepcopy(self.state)
                self.state['table_fields'][field] = [] if field == 'fields' else 'OTHER'
                with self.assertRaises(Exception): self.normalize()
                self.state = original

    def test_other_native_bindings_and_original_module_unchanged(self):
        self.assertEqual(pool_runtime._binding('field_name', self.record, self.state), '$metadata.field_name')
        self.assertIsNone(pool_runtime._binding('table_fields', self.record, self.state))

if __name__ == '__main__': unittest.main()
