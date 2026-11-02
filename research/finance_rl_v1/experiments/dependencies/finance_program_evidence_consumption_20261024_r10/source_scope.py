"""Exact frozen R9 source provider, no lookup policy or data changes."""
import importlib.util
from pathlib import Path
path=Path(__file__).resolve().parent.parent/'finance_program_recovery_source_scope_20261024_r9/source_scope.py'
spec=importlib.util.spec_from_file_location('exact_r9_private_source_provider',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
SourceScopedMetadata=module.SourceScopedMetadata
