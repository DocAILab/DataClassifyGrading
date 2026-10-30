"""Only already validated library code, never old supervision or adapters."""
import importlib.util
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
B80=ROOT.parent/'finance_program_student_b80_20261021_r3'
PREP=ROOT.parent/'finance_program_student_merge162_local_20261022_r1'
for path in (B80,PREP):
    if str(path) not in sys.path:sys.path.append(str(path))
import training_contract as identity
import dataset_contract as encoding
spec=importlib.util.spec_from_file_location('validated_training_math_only',B80/'train_student.py')
math_runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(math_runtime)
from memory_runtime import configure_gradient_checkpointing
