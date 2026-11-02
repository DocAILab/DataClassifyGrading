import importlib.util
from common import CAMPAIGN
spec=importlib.util.spec_from_file_location('same_native_runner',CAMPAIGN/'runner.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
if __name__=='__main__':raise SystemExit(module.main())
