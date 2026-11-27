"""Pinned R4 scientific helpers with an additive R6 output directory."""
import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'finance_teacher_program_coldstart_20261024_r4'
spec=importlib.util.spec_from_file_location('R6_pinned_R4_campaign',PARENT/'campaign.py');old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old)
for name in dir(old):
 if not name.startswith('_') and name!='ROOT':globals()[name]=getattr(old,name)
ROOT=Path(__file__).resolve().parent
# Imported helper globals retain pinned R4 constants. They do not write outputs.
