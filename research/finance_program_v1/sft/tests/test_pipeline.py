import tempfile
import unittest
from pathlib import Path
try:
    from pipeline import claim_launch, formal_gate
except ImportError:
    claim_launch = formal_gate = None


class PipelineTests(unittest.TestCase):
    def test_launch_claim_is_exclusive(self):
        self.assertIsNotNone(claim_launch, 'missing unique launch gate')
        with tempfile.TemporaryDirectory() as d:
            claim_launch(Path(d), 'sig')
            with self.assertRaises(FileExistsError):
                claim_launch(Path(d), 'sig')

    def test_nonzero_trial_exit_blocks_formal(self):
        self.assertIsNotNone(formal_gate, 'missing stage gate')
        with self.assertRaises(ValueError):
            formal_gate({'exit_code': 1}, None, {})

    def test_zero_trial_exit_without_proof_blocks_formal(self):
        self.assertIsNotNone(formal_gate, 'missing stage gate')
        with self.assertRaises(ValueError):
            formal_gate({'exit_code': 0}, None, {})


if __name__ == '__main__':
    unittest.main()
