"""Preserve the sealed public recovery projection without modification."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_recovery_feedback_20261023_r3/feedback.py'
spec=importlib.util.spec_from_file_location('evidence_consumption_original_projection',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
project=module.project
