"""Unchanged B80 generation; only the separately locked stage name differs."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_skill_recovery20_20261021_r5/backend.py'
spec=importlib.util.spec_from_file_location('same_finite48k_backend',path)
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
previous.valid_stage=lambda admission,deadline:(admission.get('evaluation_stage')=='skill-recovery20-contract-projection-r9'
    and admission.get('training_gpu_deadline_unix')==deadline==1792575276.1708825)
backend_class=previous.backend_class
