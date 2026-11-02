"""Exact R5 interface ablation; full native source evidence and gaps are retained."""
import importlib.util
from common import ROOT
path=ROOT.parent/'finance_program_recovery_feedback_20261024_r5/interface.py'
spec=importlib.util.spec_from_file_location('verified_r5_without_duplicate_brief',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rewrite,RecoverySession,RecoveryAdapter=module.rewrite,module.RecoverySession,module.RecoveryAdapter
