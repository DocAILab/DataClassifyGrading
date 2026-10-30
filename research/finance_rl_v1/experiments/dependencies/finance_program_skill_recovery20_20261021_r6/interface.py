"""New additive interface revision; original r5 remains immutable."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent
path = ROOT.parent / 'finance_program_skill_recovery20_20261021_r5/interface.py'
spec = importlib.util.spec_from_file_location('unchanged_r5_interface', path)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
RULE = ('Synthesis completion v3: Keep summary at most 16 words and metadata_semantics at most 100 words. '
        'State observed facts, auxiliary hypotheses, and unresolved uncertainty concisely; brevity does not '
        'permit dropping a decision-critical uncertainty or declaring it resolved. Do not repeat the same '
        'long description in both strings. Close BOTH JSON objects, the inner data object AND the outer '
        'root object, before ending the answer. The answer must end with two closing braces after the '
        'metadata_semantics string: {"summary":"brief observed context","data":{"metadata_semantics":'
        '"Observed: supplied facts; hypotheses: qualified interpretations; unresolved: actual gaps."}}. '
        'Replace the example with supplied evidence. No Markdown, extra keys, category or grade; '
        'do not invent evidence or convert hypotheses into raw facts.')

def rewrite(messages):
    result = previous.rewrite(messages)
    if result[0]['content'].startswith('Execute only the selected Skill SYNTHESIZE_METADATA_SEMANTICS.'):
        result[0]['content'] += '\n' + RULE
    return result
