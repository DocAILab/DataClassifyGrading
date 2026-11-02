"""Reuse pinned original clock/identity backend, no new inference behavior."""
import importlib.util
from pathlib import Path
s=importlib.util.spec_from_file_location('same20_clock_backend',Path(__file__).resolve().parent.parent/'finance_program_skill_recovery20_20261021_r1/backend.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
backend_class=m.backend_class
valid_stage=m.valid_stage
