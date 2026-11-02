import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r11/source_scope.py'
spec=importlib.util.spec_from_file_location('n100_exact_source_scope',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
SourceScopedMetadata=module.SourceScopedMetadata
