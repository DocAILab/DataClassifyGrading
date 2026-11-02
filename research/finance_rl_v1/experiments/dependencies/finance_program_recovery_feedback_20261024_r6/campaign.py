import importlib.util
from common import PREVIOUS
spec=importlib.util.spec_from_file_location('same_bounded_clock',PREVIOUS/'campaign.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
resume_admission=module.resume_admission
