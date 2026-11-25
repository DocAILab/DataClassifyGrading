"""Keep the exact native 16 Program/32 Skill/10 Planner limits."""
import importlib.util
from pathlib import Path
path = Path(__file__).resolve().parent.parent / 'finance_program_skill_recovery20_20261021_r1/runtime.py'
spec = importlib.util.spec_from_file_location('pinned_wide20_runtime', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
build, limits_remaining = module.build, module.limits_remaining
