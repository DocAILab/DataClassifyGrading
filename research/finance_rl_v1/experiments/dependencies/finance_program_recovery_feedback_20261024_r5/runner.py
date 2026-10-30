"""Reuse verified native execution, with current common/interface/backend modules."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_recovery_feedback_20261024_r4/runner.py'
spec=importlib.util.spec_from_file_location('same_native_runner',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=='__main__':raise SystemExit(module.main())
