"""Reuse unchanged 16/32/10 native runtime; not a six-step ceiling."""
import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('same20_wide_runtime',Path(__file__).resolve().parent.parent/'finance_program_skill_recovery20_20261021_r1/runtime.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
build=m.build
limits_remaining=m.limits_remaining
