"""Same SFT backend, only the finite422 call-accounting limit changes."""
from support import BASE
source=(BASE/'live_backend.py').read_text(encoding='utf-8')
assert source.count('if index > 4200:')==1
source=source.replace('if index > 4200:','if index > 17724:')
source=source.replace('100 cases times 42 maximum model calls exhausted','422 cases times 42 maximum model calls exhausted')
exec(compile(source,'<full422-identical-SFT-backend-new-cohort-budget>','exec'),globals())
