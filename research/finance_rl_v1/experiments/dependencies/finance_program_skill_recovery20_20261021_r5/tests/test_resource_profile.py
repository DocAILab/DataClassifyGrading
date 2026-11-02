import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
candidate = ROOT / 'resource_profile.py'
if not candidate.exists():
    candidate = ROOT.parent / 'finance_program_skill_context_contract_20261021_r5/resource_profile.py'
spec = importlib.util.spec_from_file_location('selected_two_gpu_profile', candidate)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
GIB = 1024 ** 3
UUIDS = {0: 'GPU-4e00b91b-5123-d78d-35f3-e72212f3f588',
         2: 'GPU-4c58bc24-4fe1-36cb-55ce-e43f346d954c'}

def rows():
    return [dict(index=i, uuid=UUIDS.get(i, 'GPU-other'), total_bytes=32*GIB,
                 free_bytes=23*GIB, bf16_compatible=True, other_compute_pids=[123])
            for i in (0, 1, 2)]

class SelectedResourceTests(unittest.TestCase):
    def test_only_exact_user_pair_in_requested_order(self):
        got = profile.candidates(rows(), max_sequence_tokens=49152)
        self.assertEqual([c['gpu_indices'] for c in got], [[2, 0]])
        self.assertEqual(got[0]['allocator_limit_bytes'], [20*GIB]*2)
        self.assertEqual(got[0]['scientific_updates_allowed'], 0)
        self.assertEqual(profile.recheck(rows(), got[0])['gpu_uuids'], [UUIDS[2], UUIDS[0]])

    def test_actual_two_gib_headroom_is_not_waived(self):
        observed = rows()
        observed[0]['free_bytes'] = 22*GIB - 1
        self.assertEqual(profile.candidates(observed, max_sequence_tokens=49152), [])

    def test_index_reuse_does_not_select_another_uuid(self):
        observed = rows()
        observed[2]['uuid'] = 'GPU-replaced'
        self.assertEqual(profile.candidates(observed, max_sequence_tokens=49152), [])

if __name__ == '__main__':
    unittest.main()
