"""Frozen P14/F66 interfaces and one-time separate contract installation."""
from pathlib import Path
from dataclasses import replace
import json
from agent.orchestration.catalog import SkillSpec, default_catalog, dynamic_loop_catalog

ROOT = Path(__file__).resolve().parent
REFERENCE = json.loads((ROOT/'skill_reference.json').read_text(encoding='utf-8'))
NAMES = json.loads((ROOT/'reference_skills_63.json').read_text(encoding='utf-8'))
SHARED = dynamic_loop_catalog(include_field_metadata=True, include_description_skills=True, two_stage=True)
SEARCH_NAMES = ('SEARCH_STANDARD','SEARCH_STANDARD_BY_FIELD','SEARCH_STANDARD_BY_CONTEXT','SEARCH_STANDARD_BY_TABLE','SEARCH_STANDARD_BY_DATABASE','RESEARCH_STANDARD')
OUTPUTS = {
    'GET_FIELD_DESCRIPTION':'field_description','GET_TABLE_DESCRIPTION':'table_description',
    'GET_DATABASE_DESCRIPTION':'database_description','GET_TABLE_FIELDS':'table_fields',
    'GET_CONTEXT_FIELD_DESCRIPTIONS':'context_field_descriptions','GET_CONTEXT_FIELD_TYPES':'context_field_types',
    'GET_CONTEXT_FIELD_VALUES':'context_field_values','SELECT_CONTEXT_FIELDS':'selected_context_fields',
    'EXPAND_SELECTED_CONTEXT':'context_fields','GET_COMPLETE_TABLE_CONTEXT':'table_context',
    'GET_COMPLETE_DATABASE_CONTEXT':'database_context','ACQUIRE_MISSING_METADATA':'acquired_metadata',
    'INTERPRET_FIELD_TYPE':'field_type_semantics','INTERPRET_FIELD_VALUE':'value_semantics',
    'EXPAND_TABLE_ABBREVIATION':'table_abbreviation_semantics','EXPAND_DATABASE_ABBREVIATION':'database_abbreviation_semantics',
    'INTERPRET_DATABASE_NAME':'database_semantics','INTERPRET_DATABASE_DESCRIPTION':'database_description_semantics',
    'REFINE_STANDARD_QUERY':'refined_semantic_query','SUMMARIZE_STANDARD_EVIDENCE':'standard_evidence_summary',
    'FUSE_FIELD_EVIDENCE':'field_evidence','FUSE_CONTEXT_EVIDENCE':'context_evidence',
    'FUSE_TABLE_EVIDENCE':'table_evidence','FUSE_DATABASE_EVIDENCE':'database_evidence',
    'FUSE_METADATA_EVIDENCE':'metadata_evidence','FUSE_STANDARD_EVIDENCE':'standard_evidence',
    'WEIGH_EVIDENCE':'evidence_weights','RESOLVE_FIELD_CONTEXT_CONFLICT':'field_context_resolution',
    'RESOLVE_METADATA_CONFLICT':'metadata_conflict_resolution','RESOLVE_STANDARD_CONFLICT':'standard_conflict_resolution',
    'ASSESS_CATEGORY_SUPPORT':'category_support','COMPARE_CATEGORY_HYPOTHESES':'category_comparison',
    'IDENTIFY_MISSING_EVIDENCE':'missing_evidence','DECIDE_NEXT_EVIDENCE_SOURCE':'next_evidence_source',
    'VERIFY_CATEGORY_WITH_STANDARD':'category_verification',
    'VERIFY_CLASSIFICATION_GRADING_CONSISTENCY':'grading_consistency',
}


def _inputs(text):
    return () if text in ('','无') else tuple(v.strip() for v in text.split(','))


def pool_catalog(pool: str):
    if pool not in ('P14','F66'):
        raise ValueError('pool must be P14 or F66')
    if pool=='P14':
        return dict(SHARED)
    legacy=default_catalog()
    result={}
    for role,names in NAMES.items():
        if not isinstance(names,list):
            continue
        for name in names:
            if name in SHARED:
                result[name]=SHARED[name]
            elif name in legacy:
                result[name]=legacy[name]
                if name=='SYNTHESIZE_METADATA_SEMANTICS':
                    result[name]=replace(result[name],optional_inputs=_inputs(REFERENCE[name]['optional']))
            else:
                ref=REFERENCE[name]
                produced='standard_candidates' if name in SEARCH_NAMES else OUTPUTS.get(name,name.lower())
                result[name]=SkillSpec(name,role,_inputs(ref['required']),_inputs(ref['optional']),(produced,))
    result.update(SHARED)
    return result


def purpose(name):
    if name in REFERENCE:
        return REFERENCE[name]['purpose']
    return {
        'GET_FIELD_METADATA':'Retrieve actual allowlisted metadata for the privately identified target record.',
        'SELECT_CATEGORY_TOPK':'Rank exactly five actual frozen Registry options by supplied evidence.',
        'INFER_CATEGORY_FROM_TOPK':'Select one actual prior Top-5 option and bind its frozen standard entry.'
    }[name]


def install_contracts(skill_root):
    """Explicit pre-freeze action; creates missing contracts, never edits existing files."""
    destination=Path(skill_root)/'full_pool'
    existing=[p for p in (ROOT/'runtime/skills').rglob('*.md') if 'full_pool' not in p.parts]
    patches=[]
    for name,spec in pool_catalog('F66').items():
        target=destination/f'{spec.role}_{name}.md'
        if target.exists():
            continue
        source=next((p for p in existing if p.name==target.name and ('neighbor_extra' not in p.parts)),None)
        if source is not None and name!='SYNTHESIZE_METADATA_SEMANTICS':
            content=source.read_bytes().decode('utf-8')
        else:
            header='\n'.join([
                '---',f'name: {name}',f'dsl: {spec.role}',
                f'required_inputs: [{", ".join(spec.required_inputs)}]',
                f'optional_inputs: [{", ".join(spec.optional_inputs)}]',
                f'produces: [{", ".join(spec.produces)}]',f'terminal: {str(spec.terminal).lower()}','---','',
            ])
            ref=REFERENCE[name]
            body=f'# {name}\n\nPurpose: {purpose(name)}\n\nObservation: {ref["output"]}\n\n'
            boundaries={
                'metadata_acquisition':'Use deterministic canonical Val access only. Verify private record identity. Never invent missing descriptions, types, values or locators; explicitly report unavailable data. Context belongs to the actual domain/database/table and has bounded, disclosed coverage.',
                'semantic_interpretation':'Interpret only supplied evidence. Separate direct evidence, auxiliary evidence, and uncertain hypotheses. Do not retrieve standards or predict category/grade. Never turn inferred semantics into raw metadata.',
                'standard_alignment':'Use only the actual frozen standard candidates and their disclosed descriptions. Explain support, contradictions, auxiliary context and unresolved distinctions. Never invent standard entries or overwrite canonical candidate/category/grade state.',
                'evidence_reasoning':'Use only supplied evidence and actual standards. Explain agreement, conflict, missing evidence and relative strength. Do not invent metadata, standard entries, categories or grades, or overwrite normative state.'
            }
            body+=boundaries[spec.role]+'\n\nReturn exactly {"summary": "nonempty natural-language observation", "data": {...}}. '
            body+=f'Data keys must be exactly {json.dumps(spec.produces)}. '
            body+='Semantic and reasoning outputs are nonempty strings; metadata/context/candidate outputs retain their typed validated structures. No extra keys.\n'
            content=header+body
        patches.append((target,content))
    # Runtime contract installation is an explicit artifact generation operation.
    destination.mkdir(parents=True,exist_ok=True)
    for target,content in patches:
        with target.open('xb') as stream:
            stream.write(content.encode('utf-8'))
    return destination
