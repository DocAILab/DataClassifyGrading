import unittest
from unittest.mock import patch
from watch import live


class WatchTests(unittest.TestCase):
    def test_exiting_process_is_not_a_watcher_failure(self):
        with patch('pathlib.Path.exists',return_value=True),patch('pathlib.Path.read_bytes',side_effect=ProcessLookupError('process exited')):
            try:value=live(123,'train_student.py')
            except ProcessLookupError:self.fail('process exit during polling must return false, not abort watcher')
        self.assertFalse(value)
