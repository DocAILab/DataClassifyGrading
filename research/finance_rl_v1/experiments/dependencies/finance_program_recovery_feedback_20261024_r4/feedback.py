"""Load the sealed r3 public-only projection without modifying it."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_recovery_feedback_20261023_r3/feedback.py'
spec=importlib.util.spec_from_file_location('sealed_r3_feedback',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
project=module.project
