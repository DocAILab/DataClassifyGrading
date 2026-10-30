"""Same B80/48K backend, explicitly authorized new bounded inference stage."""
import importlib.util
from pathlib import Path
path = Path(__file__).resolve().parent.parent / 'finance_program_skill_recovery20_20261021_r5/backend.py'
spec = importlib.util.spec_from_file_location('same_b80_finite48k_backend', path)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
previous.valid_stage = lambda admission, deadline: (admission.get('evaluation_stage') == 'skill-recovery20-taskscope-r8'
    and admission.get('training_gpu_deadline_unix') == deadline == 1792575276.1708825)
backend_class = previous.backend_class
