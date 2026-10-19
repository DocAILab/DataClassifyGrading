import json
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
import zipfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from artifacts import BUDGET, GENERATION, code_inventory, load_controller, load_study, put, read, sha
from backends import ScriptedBackend
from bundle import bundle
from prepare import project, registry_from, gold_for
from replay import replay
from runner import exclusive_run, run
from score import score_traces, score_run


def corpus():
    return [{'id': f'std-{i}', 'category_root_level': 'synthetic',
             'category_leaf_level': 'same' if i < 3 else f'leaf-{i}',
             'sensitivity_level': 'L2' if i == 1 else 'L1',
             'data_description_and_example': f'fictional standard {i}',
             'reference_standard': 'fictional reference'} for i in range(7)]


def raw(ident='fixture-1'):
    return {'id': ident, 'domain': 'synthetic', 'metadata': {
        'field_name': 'OPAQUE', 'field_description': 'fictional field description',
        'table_name': 'FORBIDDEN_TABLE', 'gold': 'SECRET_GOLD'},
        'classification': {'category_root_level': 'synthetic', 'category_leaf_level': 'same'},
        'grading': {'sensitivity_level': 'L1'}, 'data_description_and_example': 'SECRET_GOLD',
        'reference_standard': 'SECRET_GOLD'}


def fixture(path, count=3):
    rows = [raw(f'fixture-{n}') for n in range(count)]
    registry = registry_from(corpus())
    study = {'expected_ids': [r['id'] for r in rows], 'rows': [project(r) for r in rows],
             'registry': registry, 'skill_pool': 'F66', 'budget': BUDGET, 'generation': GENERATION}
    put(path / 'inference/study.json', study)
    put(path / 'evaluation/gold.json', {'answers': gold_for(rows, registry)})
    put(path / 'manifest.json', {'code': code_inventory(), 'study_sha256': sha(path / 'inference/study.json'),
                               'gold_sha256': sha(path / 'evaluation/gold.json')})
    return study


class VehicleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = self.root / 'package'
        self.study = fixture(self.package)
        self.output = self.root / 'run'

    def test_one_to_many_registry_and_gold(self):
        self.assertEqual(len(self.study['registry']), 7)
        self.assertEqual(len(gold_for([raw()], self.study['registry'])['fixture-1']['compatible_standard_ids']), 2)
        changed = raw(); changed['grading']['sensitivity_level'] = 'L2'
        self.assertEqual(gold_for([changed], self.study['registry'])['fixture-1']['compatible_standard_ids'], ['std-1'])

    def test_gold_sentinel_does_not_change_actual_requests(self):
        builder = load_controller()
        transcripts = []
        for secret in ('SECRET_A', 'SECRET_B'):
            source = raw(); source['classification'] = {'arbitrary': secret}
            source['reference_standard'] = secret; source['metadata']['gold'] = secret
            row = project(source)
            messages = []
            def generate(value):
                messages.append(value)
                return ScriptedBackend()(value)
            trace = builder(generate, self.study['registry'], [row], BUDGET, skill_pool='F66').run(
                {'source_id': row['id'], 'domain': row['domain'], 'field_name': row['metadata']['field_name']})
            self.assertTrue(trace.terminated)
            text = json.dumps(messages)
            self.assertNotIn(secret, text)
            self.assertNotIn('FORBIDDEN_TABLE', text)
            transcripts.append(messages)
        self.assertEqual(*transcripts)

    def test_metadata_lookup_does_not_invent_table_context(self):
        load_controller()
        from pool_metadata import PoolMetadata
        from agent.orchestration.contracts import MetadataRecord
        row = self.study['rows'][0]
        lookup = PoolMetadata([row])
        record = MetadataRecord.from_mapping({'source_id': row['id'], 'domain': row['domain'], 'field_name': 'OPAQUE'})
        context = lookup.table_fields(record)
        self.assertEqual(context['fields'], [])
        self.assertEqual(context['missing_locators'], ['database_name', 'table_name'])

    def test_unknown_program_is_bounded_and_not_autofinalized(self):
        calls = []
        def invalid(messages):
            calls.append(messages)
            return '{"program":[{"tool":"NONEXISTENT","arguments":{}}]}'
        row = self.study['rows'][0]
        trace = load_controller()(invalid, self.study['registry'], self.study['rows'], BUDGET, skill_pool='F66').run(
            {'source_id': row['id'], 'domain': row['domain'], 'field_name': 'OPAQUE'})
        self.assertFalse(trace.terminated)
        self.assertEqual(len(calls), BUDGET['max_planner_attempts'])

    def test_run_does_not_open_gold(self):
        original = Path.open
        def guard(path, *args, **kwargs):
            if path.name == 'gold.json':
                raise AssertionError('inference opened gold')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guard):
            result = run(self.package, self.output)
        self.assertEqual(result['committed_count'], 3)
        self.assertEqual(result['terminated_count'], 3)
        self.assertFalse(result['quality_claim'])
        self.assertEqual(replay(self.package, self.output)['completed_sources_replayed'], 3)

    def test_resume_reuses_exact_artifacts_and_does_not_initialize_when_done(self):
        first = run(self.package, self.output, max_samples=1)
        self.assertEqual(first['pending_count'], 2)
        original_sha = sha(self.output / 'sources/0001/receipt.json')
        result = run(self.package, self.output, resume=True)
        self.assertEqual(result['pending_count'], 0)
        self.assertEqual(original_sha, sha(self.output / 'sources/0001/receipt.json'))
        def forbidden():
            raise AssertionError('backend initialized for completed run')
        self.assertEqual(run(self.package, self.output, resume=True, factory=forbidden)['committed_count'], 3)

    def test_interrupted_source_is_preserved_and_not_rerolled(self):
        run(self.package, self.output, max_samples=1)
        folder = self.output / 'sources/0002'
        put(folder / 'started.json', {'partial': True})
        partial_sha = sha(folder / 'started.json')
        run(self.package, self.output, resume=True)
        self.assertEqual(read(folder / 'receipt.json')['status'], 'interrupted')
        self.assertEqual(sha(folder / 'started.json'), partial_sha)
        self.assertFalse((folder / 'trace.json').exists())
        score = score_run(self.package, self.output)
        self.assertEqual(score['denominator'], 3)
        self.assertFalse(score['per_id']['fixture-1']['terminated'])

    def test_saved_trace_tampering_rejected(self):
        run(self.package, self.output, max_samples=1)
        path = self.output / 'sources/0001/trace.json'
        path.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'receipt'):
            run(self.package, self.output, resume=True)

    def test_changed_study_rejected_before_backend(self):
        path = self.package / 'inference/study.json'
        path.write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'study identity'):
            run(self.package, self.output)

    def test_changed_run_identity_rejected(self):
        run(self.package, self.output, max_samples=1)
        path = self.output / 'run.json'
        value = read(path); value['backend'] = 'local_official_base'
        path.write_text(json.dumps(value), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'run identity'):
            run(self.package, self.output, resume=True)

    def test_backend_failure_halts_and_remaining_sources_keep_denominator(self):
        def failing(messages):
            raise RuntimeError('synthetic backend failure')
        result = run(self.package, self.output, factory=lambda: failing)
        self.assertEqual(result['committed_count'], 1)
        self.assertEqual(result['model_calls'], 1)
        self.assertEqual(result['halted'], 'backend_error')
        score = score_run(self.package, self.output)
        self.assertEqual(score['denominator'], 3)
        self.assertEqual(score['joint_accuracy'], 0.)
        self.assertEqual(run(self.package, self.output, resume=True)['error_count'], 1)

    def test_score_checks_grade_against_selected_standard(self):
        ids = self.study['expected_ids']
        answers = read(self.package / 'evaluation/gold.json')['answers']
        def trace(key, level):
            return {'terminated': True, 'stop_reason': 'validated_terminal',
                    'final_result': {'category': key, 'grading_level': level}}
        result = score_traces(ids, {ids[0]: trace('std-2', 'L1'), ids[1]: trace('std-1', 'L2'),
                                   ids[2]: trace('std-1', 'L1')}, answers, self.study['registry'])
        self.assertEqual(result['category_correct'], 2)
        self.assertEqual(result['grading_correct'], 1)
        self.assertEqual(result['joint_correct'], 1)
        self.assertEqual(result['terminated'], 2)

    def test_unlabelled_or_unknown_prediction_never_scores(self):
        ids = self.study['expected_ids']
        unknown = {'terminated': True, 'stop_reason': 'validated_terminal',
                   'final_result': {'category': 'UNKNOWN', 'grading_level': 'L1'}}
        answers = read(self.package / 'evaluation/gold.json')['answers']
        result = score_traces(ids, {ids[0]: unknown}, answers, self.study['registry'])
        self.assertEqual(result['joint_correct'], 0)
        self.assertEqual(result['denominator'], 3)
        self.assertEqual(result['missing_or_failed_traces'], 2)

    def test_run_lock_excludes_concurrent_writer(self):
        self.output.mkdir()
        with exclusive_run(self.output):
            with self.assertRaises(OSError):
                with exclusive_run(self.output):
                    pass

    def test_gold_hash_checked_only_by_scorer(self):
        run(self.package, self.output)
        (self.package / 'evaluation/gold.json').write_text('{}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'gold identity'):
            score_run(self.package, self.output)

    def test_inference_bundle_is_portable_and_excludes_gold(self):
        archive = self.root / 'deployment.zip'
        bundle(self.package, archive)
        extracted = self.root / 'deployed'
        with zipfile.ZipFile(archive) as z:
            self.assertNotIn('package/evaluation/gold.json', z.namelist())
            self.assertEqual({n for n in z.namelist() if n.startswith('package/')},
                             {'package/manifest.json', 'package/inference/study.json'})
            z.extractall(extracted)
        completed = subprocess.run([sys.executable, '-B', str(extracted / 'code/runner.py'),
                                    '--package', str(extracted / 'package'), '--output', str(self.root / 'deployed-run'),
                                    '--backend', 'mock'], capture_output=True, text=True, timeout=30)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)['committed_count'], 3)

    def test_top5_keeps_standard_identity_and_macro_counts_unsupported_predictions(self):
        ids = self.study['expected_ids']
        answers = read(self.package / 'evaluation/gold.json')['answers']
        trace = {'terminated': True, 'stop_reason': 'validated_terminal',
                 'final_result': {'category': 'std-3', 'grading_level': 'L1'},
                 'programs': [{'steps': [{'status': 'success', 'call': {'tool': 'SELECT_CATEGORY_TOPK'},
                     'data': {'standard_candidates': [{'category': f'std-{i}'} for i in range(5)]}}]}]}
        value = score_traces(ids, {ids[0]: trace}, answers, self.study['registry'])
        self.assertEqual(value['top5_first_hit'], 1)
        self.assertEqual(value['top5_compatible_entry_any_hit'], 1)
        self.assertEqual(value['category_correct'], 0)
        self.assertEqual(value['registry_leaf_paths'], 5)


if __name__ == '__main__':
    unittest.main()
