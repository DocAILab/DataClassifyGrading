"""One prompt-only type alignment, no raw output conversion or native relaxation."""
from copy import deepcopy
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('frozen_r4_context_interface',ROOT.parent/'finance_program_skill_recovery20_20261021_r4/interface.py')
previous=importlib.util.module_from_spec(s);s.loader.exec_module(previous)

CONTEXT_RULE=('Context interpretation output contract: context_field_semantics MUST be one nonempty JSON string, '
 'not an array or object. If you explain multiple fields, put their explanations together inside that ONE string. '
 'Do not use a list of field_name/semantics objects or create additional data keys. '
 'Return exactly one JSON object with only summary and data; summary is one nonempty string; '
 'data has only context_field_semantics. The exact shape is '
 '{"summary":"brief source-supported overview","data":{"context_field_semantics":"One text explaining the supplied neighboring fields, their observed relationship, and unresolved uncertainty."}}. '
 'Replace example prose with the actual supplied observations. Keep quotes escaped as required by JSON. '
 'Use only resolved inputs; preserve all original evidence-boundary rules. Do not infer category/grade, '
 'invent values, certify model hypotheses as raw facts, or alter original metadata.')

def rewrite_actual(messages,stage):
    result=deepcopy(messages)
    if stage=='INTERPRET_CONTEXT_FIELDS':
        if not result[0]['content'].startswith('Execute only the selected Skill INTERPRET_CONTEXT_FIELDS.'):
            raise ValueError('target Skill/system stage mismatch')
        result[0]['content']+='\n'+CONTEXT_RULE
    return result

def rewrite(messages):
    result=previous.rewrite(messages)
    stage='INTERPRET_CONTEXT_FIELDS' if result[0]['content'].startswith('Execute only the selected Skill INTERPRET_CONTEXT_FIELDS.') else 'unchanged'
    return rewrite_actual(result,stage)
