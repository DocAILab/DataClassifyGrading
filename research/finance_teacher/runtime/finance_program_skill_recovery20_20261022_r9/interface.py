"""Project existing public contracts; never edit a model output or native state."""
import importlib.util
import json
from pathlib import Path
import re
path = Path(__file__).resolve().parent.parent / 'finance_program_skill_recovery20_20261022_r8/interface.py'
spec = importlib.util.spec_from_file_location('previous_taskscope_interface', path)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)

def rewrite(messages):
    actual = previous.rewrite(messages)
    system = actual[0]['content']
    stage = re.match(r'Execute only the selected Skill ([A-Z_]+)\.', system)
    if stage is None:
        # Read the original public request, before reversible message encoding.
        body = json.loads(messages[-1]['content'])
        feedback = body.get('feedback') or {}
        error = feedback.get('error') or ''
        missing = re.fullmatch(r'step \d+: missing dependency ([A-Za-z_]+)', error)
        if missing:
            key = missing.group(1)
            producers = [{'name':s['name'], 'required_inputs':s['required_inputs'],
                          'produces':s['produces']}
                         for s in body.get('skill_library', []) if key in s.get('produces', [])]
            actual[0]['content'] += ('\nThe latest plan was rejected before execution. '
                'Do not repeat the same missing input. Descriptions and names have distinct output keys. '
                'The following is copied from public contracts, not a prescribed plan or a guarantee '
                'that a producer is currently executable. Check its actual required inputs; choose '
                'a valid remaining plan yourself. No automatic steps or new observations are supplied.'
                '\nLocal dependency projection (not an executed plan):\n' + json.dumps(dict(
                    missing_dependency=key, declared_producers=producers, evidence_added=False),
                    ensure_ascii=False, separators=(',',':')))
        return actual
    name = stage.group(1)
    if name in {'INFER_CATEGORY', 'INFER_CATEGORY_FROM_TOPK'}:
        examples = [dict(option_id='<one supplied option_id>', summary='<brief supported conclusion>',
            support=dict(sufficient=True, citations=[dict(fact_id='<exact supplied eligible fact_id>',
            quote='<exact supporting source text>')], missing_evidence='')),
            dict(option_id='UNRESOLVED', summary='<brief unresolved distinction>',
            support=dict(sufficient=False, citations=[], missing_evidence='<actual missing distinction>'))]
        actual[0]['content'] += ('\nBoth sufficient and insufficient branches MUST include all three '
            'support keys; missing_evidence is mandatory even when empty. Select a branch by actual '
            'evidence, never by this example. Use only exact fact IDs from evidence_boundary.facts; '
            'an array-index path is not interchangeable with a canonical fact ID. Do not copy '
            'placeholders or manufacture a decisive fact. Original sufficiency rules remain in force.'
            '\nComplete output shapes (structure only; replace every placeholder):\n' +
            json.dumps(examples, ensure_ascii=False, separators=(',',':')))
    elif name in {'SYNTHESIZE_METADATA_SEMANTICS', 'INTERPRET_CONTEXT_FIELDS'}:
        key = {'SYNTHESIZE_METADATA_SEMANTICS':'metadata_semantics',
               'INTERPRET_CONTEXT_FIELDS':'context_field_semantics'}[name]
        example = dict(summary='<brief actual observation>', data={key:
            '<Observed: supplied facts; Hypotheses: uncertain interpretations; Unresolved: actual gaps>'})
        actual[0]['content'] += ('\nKeep each string concise without omitting decisive uncertainty. '
            'Do not repeat the same paragraph in both strings. The last two characters must close '
            'the data object AND the root object: }}. End only after both objects are closed. '
            'Do not infer a category, grade, missing source fact or semantic certainty from this shape.'
            '\nComplete output shape (structure only; replace every placeholder):\n' +
            json.dumps(example, ensure_ascii=False, separators=(',',':')))
    elif name == 'COMPARE_STANDARD_CANDIDATES':
        actual[0]['content'] += ('\nReturn a concise reason string, then close the root object before '
            'ending. The exact existing flat shape is {"decision":"no_match","option_id":null,'
            '"reason":"<actual unresolved comparison>"}; use match and an actually supplied option '
            'only when supported. Never copy an unsupported decision from this structural example.')
    return actual
