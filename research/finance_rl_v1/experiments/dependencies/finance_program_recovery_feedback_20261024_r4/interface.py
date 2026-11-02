"""Per-rollout r3 session, unchanged semantic/terminal contracts."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_recovery_feedback_20261023_r3/interface.py'
spec=importlib.util.spec_from_file_location('sealed_r3_interface',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
RecoveryAdapter=module.RecoveryAdapter
RecoverySession=module.RecoverySession
rewrite=module.rewrite
