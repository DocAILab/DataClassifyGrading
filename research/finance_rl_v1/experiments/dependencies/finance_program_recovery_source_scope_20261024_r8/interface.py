"""Retain r6 prompt and per-case session exactly; only retrieval is changed."""
import importlib.util
from common import PREVIOUS
spec=importlib.util.spec_from_file_location('unchanged_r6_source_brief_interface',PREVIOUS/'interface.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
rewrite,RecoverySession,RecoveryAdapter=module.rewrite,module.RecoverySession,module.RecoveryAdapter
