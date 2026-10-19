"""Scripted CPU fixture and opt-in offline local base model; no adapters."""
import json
from pathlib import Path
import sys
from artifacts import ROOT, MODEL_REVISION, read


class ScriptedBackend:
    kind = 'scripted_cpu_test'

    def __call__(self, messages):
        head = messages[0]['content'].splitlines()[0]
        user = json.loads(messages[-1]['content'])
        if head.startswith('You are the Global Program Planner'):
            names = ['GET_FIELD_METADATA', 'INTERPRET_FIELD_NAME', 'SELECT_CATEGORY_TOPK',
                     'INFER_CATEGORY_FROM_TOPK', 'INFER_GRADING_LEVEL', 'FINALIZE_CLASSIFICATION']
            return json.dumps({'program': [{'tool': n, 'arguments': {}} for n in names]})
        skill = head.removeprefix('Execute only the selected Skill ').removesuffix('.')
        if skill == 'INTERPRET_FIELD_NAME':
            ctx = user['grounding_context']
            desc = ctx.get('field_description')
            status = 'observed_description' if desc else 'unknown'
            return json.dumps({'summary': status, 'data': {'field_semantics': {
                'schema': 'grounded-source-v1', 'status': status,
                'field_name': ctx['field_name'], 'description': desc, 'source_consistency': 'not_adjudicated'}}})
        if skill == 'SELECT_CATEGORY_TOPK':
            return json.dumps({'option_ids': [r['option_id'] for r in user['category_options'][:5]], 'summary': 'scripted'})
        if skill == 'INFER_CATEGORY_FROM_TOPK':
            return json.dumps({'option_id': user['candidate_details'][0]['option_id'], 'summary': 'scripted'})
        raise ValueError('unexpected scripted role')


def local_backend(model, folder):
    """The unchanged team FP16 greedy backend loads only an already verified local model."""
    manifest = read(Path(model) / 'model-manifest.json')
    if manifest['repo_id'] != 'Qwen/Qwen3.5-9B' or manifest['revision'] != MODEL_REVISION:
        raise ValueError('base model revision differs')
    sys.path.insert(0, str(ROOT / 'upstream/backend'))
    from local_smoke import Backend
    return Backend(model, folder)
