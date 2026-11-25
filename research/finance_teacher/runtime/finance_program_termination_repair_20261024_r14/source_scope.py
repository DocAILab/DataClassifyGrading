import importlib.util
from pathlib import Path
p=Path(__file__).resolve().parent.parent/'finance_program_evidence_consumption_20261024_r11/source_scope.py'
s=importlib.util.spec_from_file_location('v13_exact_source_scope',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
SourceScopedMetadata=m.SourceScopedMetadata
