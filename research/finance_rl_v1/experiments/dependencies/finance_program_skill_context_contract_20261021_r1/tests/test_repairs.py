import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'finance_program_skill_recovery20_20261021_r4'
E2=ROOT.parent/'finance_program_student_b80_eval_20261021_r2'

def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

interface=load('context_contract_under_test',ROOT/'interface.py' if (ROOT/'interface.py').exists() else OLD/'interface.py')
policy=load('context_policy_under_test',ROOT/'context_policy.py') if (ROOT/'context_policy.py').exists() else None
backend_path=ROOT/'backend.py' if (ROOT/'backend.py').exists() else E2/'eval_backend.py'

class Repairs(unittest.TestCase):
    def test_explicit_string_contract_on_all_three_actual_failures(self):
        for call in [6,13,18]:
            p=OLD/f'reviews/final-01/extract/runs/native-12-unified/call-{call:03d}/model-request.json'
            original=json.loads(p.read_bytes())['messages'];saved=json.dumps(original,ensure_ascii=False)
            fn=getattr(interface,'rewrite_actual',lambda messages,stage:messages)
            new=fn(original,'INTERPRET_CONTEXT_FIELDS')
            self.assertIn('context_field_semantics MUST be one nonempty JSON string',new[0]['content'])
            self.assertIn('not an array or object',new[0]['content'])
            self.assertEqual(new[1:],original[1:]);self.assertEqual(json.dumps(original,ensure_ascii=False),saved)

    def test_48k_cap_is_finite_and_reserves_full_generation(self):
        self.assertIsNotNone(policy,'no context policy; old backend remains32768')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text(json.dumps({'text_config':{'max_position_embeddings':262144}}),encoding='utf-8')
            a={'base_model_dir':d,'max_sequence_tokens':49152,'context_window_tokens':49152}
            self.assertEqual(policy.validate_context_size(37429,a),49152)
            self.assertEqual(policy.validate_context_size(48640,a),49152)
            for n in [48641,0,-1,True,1.5]:
                with self.assertRaises(ValueError):policy.validate_context_size(n,a)
            with self.assertRaises(ValueError):policy.validate_context_size(100,dict(a,context_window_tokens=262144))
            p.write_text(json.dumps({'text_config':{'max_position_embeddings':32768}}),encoding='utf-8')
            with self.assertRaises(ValueError):policy.validate_context_size(100,a)

    def test_generation_uses_actual_new_budget_guard(self):
        source=backend_path.read_text(encoding='utf-8')
        self.assertIn('validate_context_size',source,'actual generation still uses hard32768 guard')
        self.assertIn('source.count',source,'pinned backend adaptation must reject unknown source')

    def test_other_stages_unchanged(self):
        messages=[{'role':'system','content':'Execute only the selected Skill SELECT_CONTEXT_FIELDS.\noriginal'},{'role':'user','content':'{}'}]
        fn=getattr(interface,'rewrite_actual',lambda m,s:m)
        self.assertEqual(fn(messages,'SELECT_CONTEXT_FIELDS'),messages)

    def test_memory_envelope_is_not_relaxed(self):
        if not (ROOT/'resource_profile.py').exists():self.fail('new48K resource profile missing')
        m=load('context_profile_under_test',ROOT/'resource_profile.py')
        rows=[dict(index=i,uuid=f'GPU-test{i}',free_bytes=30*1024**3,total_bytes=32*1024**3,other_compute_pids=[],bf16_compatible=True) for i in range(3)]
        choices=m.candidates(rows,max_sequence_tokens=49152);three=[v for v in choices if len(v['gpu_indices'])==3][0]
        self.assertEqual(three['allocator_limit_bytes'],[18*1024**3]*3)
        self.assertEqual(three['model_max_memory_bytes'],[8*1024**3]*3)
        self.assertEqual(three['headroom_bytes'],2*1024**3)
        self.assertEqual(m.recheck(rows,three)['max_sequence_tokens'],49152)
        with self.assertRaises(ValueError):m.candidates(rows,max_sequence_tokens=49153)

if __name__=='__main__':unittest.main()
