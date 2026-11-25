"""Latest type prompt plus unchanged reversible encoding and selector contract."""
import importlib.util
from pathlib import Path
path = Path(__file__).resolve().parent.parent / 'finance_program_skill_context_contract_20261021_r1/interface.py'
spec = importlib.util.spec_from_file_location('pinned_latest_context_interface', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rewrite = module.rewrite
