"""Only correct native-controller initialization in the targeted worker."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT.parent / 'finance_program_skill_recovery20_20261021_r6/run_smoke.py'
source = path.read_text(encoding='utf-8')
old = "    load_engine(E2)\n    from agent.orchestration.model_skill_executor import parse_skill_observation\n    skill = SimpleNamespace(name='SYNTHESIZE_METADATA_SEMANTICS', produces=('metadata_semantics',))"
assert source.count(old) == 1
source = source.replace(old, '    from native_skill import native_skill\n    parse_skill_observation, skill = native_skill()')
namespace = dict(__file__=str(ROOT / 'run_smoke.py'), __name__='single_skill_native_init_fixed')
exec(compile(source, str(path)+'[controller-initialization-only]', 'exec'), namespace)
if __name__ == '__main__':
    raise SystemExit(namespace['main']())
