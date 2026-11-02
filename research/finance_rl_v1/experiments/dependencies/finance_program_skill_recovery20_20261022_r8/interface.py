"""Task-scope clarification only; acquired facts and all native guards unchanged."""
import importlib.util
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
path = ROOT.parent / 'finance_program_skill_recovery20_20261021_r7/interface.py'
spec = importlib.util.spec_from_file_location('verified_synthesis_and_context_contract', path)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

TASK = ('Field-definition classification: Classify the semantic purpose of this database FIELD, '
        'not the content of a particular unseen row. Actually acquired field and table descriptions '
        'are raw source evidence about that purpose; do not automatically require a sampled row value '
        'or physical field type when those would not distinguish the candidate definitions. '
        'Compare the candidate definitions with ALL acquired target and same-table facts, including '
        'facts contradicting your initially preferred candidate. Finance domain alone never means '
        'financial-accounting data; project, date, status, account or code alone never proves a product '
        'or subject category. When multiple candidates remain plausible because their decisive '
        'subject/product distinction is genuinely absent, preserve insufficient and identify that '
        'specific distinction. Never guess to improve completion. Exact source citations remain '
        'mandatory for sufficient classification, and quoted text must actually support that distinction. '
        'Do not use reference labels, model interpretations or invented row values as raw evidence.')

RECOVERY = ('Acquisition-aware recovery: Before the first classification, check whether current '
            'acquired descriptions distinguish the likely candidates. If not, select a relevant '
            'unqueried acquisition whose required locators are actually present, for example same-table '
            'context when available, and consume its returned facts before deciding. This is not a '
            'mandatory extra sequence: do not acquire context when existing source facts are sufficient. '
            'No hypothetical acquisition result is evidence. Do not repeat obtained queries or unavailable '
            'inputs, reinterpret a field merely to repeat successful work, or call consumers of missing '
            'field_type/value/table_fields. Reuse valid state and plan only remaining dependencies. '
            'Do not retry a classification after an insufficient outcome without new eligible raw facts; '
            'the original persistent-gap guard is unchanged. Empty context or metadata aliases do not '
            'constitute new evidence. Remain within the original explicit global budget, and never '
            'force a final answer when the actual evidence is inadequate.')

def rewrite(messages):
    actual = previous.rewrite(messages)
    match = re.match(r'Execute only the selected Skill ([A-Z_]+)\.', actual[0]['content'])
    if match is None:
        actual[0]['content'] += '\n' + TASK + '\n' + RECOVERY
    elif match.group(1) in {'SELECT_CATEGORY_TOPK', 'SEARCH_STANDARD_BY_FIELD', 'SEARCH_STANDARD_BY_DOMAIN',
                            'INFER_CATEGORY', 'INFER_CATEGORY_FROM_TOPK'}:
        actual[0]['content'] += '\n' + TASK
    return actual
