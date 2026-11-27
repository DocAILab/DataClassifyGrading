"""Frozen full-pool study identity and independently reconstructible metrics."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import copy
from pathlib import Path

from common import fingerprint, read_json, read_jsonl, sha256, write_json

ROOT = Path(__file__).resolve().parent
ADAPTER_SKILLS = ('SELECT_CATEGORY_TOPK', 'INFER_CATEGORY_FROM_TOPK')
P14_SKILLS = frozenset(('CHECK_METADATA_AVAILABILITY','INTERPRET_DOMAIN','EXPAND_FIELD_ABBREVIATION',
    'INTERPRET_FIELD_NAME','DISAMBIGUATE_FIELD_SEMANTICS','SELECT_CATEGORY_TOPK',
    'COMPARE_STANDARD_CANDIDATES','CHECK_EVIDENCE_SUFFICIENCY','INFER_CATEGORY_FROM_TOPK',
    'INFER_GRADING_LEVEL','FINALIZE_CLASSIFICATION','GET_FIELD_METADATA',
    'INTERPRET_FIELD_DESCRIPTION','INTERPRET_TABLE_DESCRIPTION'))
BASE_CONFIG = {
    'raw_metadata_schema_version': 'initial-record-plus-acquired-raw-v2',
    'binding_rule_version': 'canonical-state-bindings-v1',
    'evidence_schema_version': 'typed-observed-evidence-v2',
    'schema': 1, 'pools': ['P14', 'F66'], 'conditions': ['base', 'sft'],
    'condition_order': 'all-base-before-any-sft',
    'selection': 'sha256(finance-followup-20261004: + source_id), ascending digest; ID tie-break',
    'pilot_size': 20, 'expanded_size': 60, 'smoke_size': 2,
    'generator_seed': 20260928,
    'generation': {'max_new_tokens': 512, 'enable_thinking': False, 'do_sample': False,
                   'eos_token_ids': [248044, 248046], 'local_files_only': True},
    'loop': {'max_turns': 8, 'max_identical_calls': 2, 'action_retries': 1,
             'forced_finalize_retries': 1, 'repetition_retries': 1,
             'recover_ready_finalization': True},
    'native_standard_backend': 'model-catalog-top5-v1',
    'adapter_skills': list(ADAPTER_SKILLS), 'new_training': False,
    'primary_recall': 'last-successful-top5',
    'metadata_source_scope': 'frozen-Val-unlabelled-only',
}

VARIANTS = {name: {'variant':name,'missing_awareness':aware,'max_turns':turns}
            for name,aware,turns in [('A',False,8),('B',True,8),('C',False,12),('D',True,12)]}

P14_ARMS = {
    'O': {'variant':'C','metadata_prefetch':False,'top1_selection_mode':'model'},
    'D': {'variant':'D','metadata_prefetch':False,'top1_selection_mode':'model'},
    'M': {'variant':'D','metadata_prefetch':True,'top1_selection_mode':'model'},
    'E': {'variant':'D','metadata_prefetch':False,'top1_selection_mode':'first'},
    'ME': {'variant':'D','metadata_prefetch':True,'top1_selection_mode':'first'},
}


def load_config(root=ROOT):
    path = Path(root)/'variant.json'
    variant = read_json(path) if path.exists() else VARIANTS['A']
    if (not isinstance(variant,dict) or set(variant) != {'variant','missing_awareness','max_turns'}
            or type(variant.get('missing_awareness')) is not bool
            or type(variant.get('max_turns')) is not int
            or variant.get('variant') not in VARIANTS
            or variant != VARIANTS[variant['variant']]):
        raise ValueError('invalid canonical variant configuration')
    config = copy.deepcopy(BASE_CONFIG)
    config.update({'variant':variant['variant'],'missing_awareness':variant['missing_awareness']})
    config['loop']['max_turns'] = variant['max_turns']
    control_path = Path(root)/'p14_controls.json'
    if control_path.exists():
        control = read_json(control_path)
        if (not isinstance(control,dict) or set(control) != {'arm','metadata_prefetch','top1_selection_mode'}
                or type(control.get('arm')) is not str or control['arm'] not in P14_ARMS
                or type(control.get('metadata_prefetch')) is not bool
                or type(control.get('top1_selection_mode')) is not str):
            raise ValueError('invalid P14 control declaration')
        expected=P14_ARMS[control['arm']]
        if (not path.is_file() or variant != VARIANTS[expected['variant']]
                or {k:v for k,v in control.items() if k!='arm'} != {k:v for k,v in expected.items() if k!='variant'}):
            raise ValueError('P14 control combination or variant differs')
        config.update({'p14_arm':control['arm'],'metadata_prefetch':control['metadata_prefetch'],
                       'top1_selection_mode':control['top1_selection_mode'],'pools':['P14'],
                       'conditions':['sft'],'condition_order':'existing-adapter-only-no-new-training'})
    return config


CONFIG = load_config()


def select_cohorts(rows):
    ids = [row.get('source_id') for row in rows]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(ids) != len(set(ids)):
        raise ValueError('source IDs must be nonempty unique strings')
    size = CONFIG['pilot_size'] + CONFIG['expanded_size'] + CONFIG['smoke_size']
    if len(ids) < size:
        raise ValueError('too few rows for disjoint Pilot, Expanded and smoke')
    ordered = sorted(ids, key=lambda sid: (hashlib.sha256(('finance-followup-20261004:'+sid).encode()).hexdigest(), sid))
    return {'pilot': ordered[:20], 'expanded': ordered[20:80], 'smoke': ordered[80:82]}


def adapter_route(condition, messages):
    if condition not in CONFIG['conditions']:
        raise ValueError('unknown condition')
    if condition == 'base' or not messages or messages[0].get('role') != 'system':
        return None
    header = messages[0].get('content', '')
    return next((name for name in ADAPTER_SKILLS
                 if header.startswith(f'Execute only the selected Skill {name}.\n')), None)


def execution_files(root):
    root = Path(root)
    paths = list(root.glob('*.py')) + list(root.glob('*.sh'))
    paths += list((root/'runtime/code').rglob('*.py'))
    paths += list((root/'runtime/skills').rglob('*.md'))
    paths += list((root/'runtime/tests').rglob('*.py'))
    paths += [path for folder in ('runtime/config', 'runtime/tests/fixtures')
              for path in (root/folder).rglob('*') if path.is_file()]
    paths += list((root/'tests').rglob('*.py'))
    for name in ('variant.json','p14_controls.json'):
        if (root/name).exists():
            paths.append(root/name)
    paths += [root/name for name in (
        'data/val.jsonl', 'data/metadata-val.json', 'data/registry.json',
        'data/split_boundaries.json', 'reference_skills_63.json', 'skill_reference.json', 'model_artifacts.json',
        'adapter_completion_reference.json')]
    if any(path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) for path in paths):
        raise ValueError('execution inventory path escape or symbolic link')
    return {path.relative_to(root).as_posix(): sha256(path) for path in sorted(set(paths))}


def freeze_study(root=ROOT):
    root = Path(root)
    if (root/'frozen/manifest.json').exists():
        raise ValueError('refuse to overwrite frozen study')
    cohorts = select_cohorts(read_jsonl(root/'data/val.jsonl'))
    payload = {'config': load_config(root), 'cohorts': cohorts, 'files': execution_files(root),
               'expected_base_manifest': read_json(root/'model_artifacts.json'),
               'adapter': read_json(root/'adapter_completion_reference.json'),
               'scope': 'development validation; no new SFT; not untouched Test'}
    manifest = {**payload, 'execution_signature': fingerprint(payload)}
    write_json(root/'frozen/manifest.json', manifest)
    return manifest


def check_study(root=ROOT):
    root = Path(root)
    manifest = read_json(root/'frozen/manifest.json')
    if fingerprint({k:v for k,v in manifest.items() if k != 'execution_signature'}) != manifest.get('execution_signature'):
        raise ValueError('frozen manifest signature mismatch')
    if manifest.get('config') != load_config(root):
        raise ValueError('frozen configuration changed')
    if manifest.get('files') != execution_files(root):
        raise ValueError('execution file inventory changed')
    if manifest.get('cohorts') != select_cohorts(read_jsonl(root/'data/val.jsonl')):
        raise ValueError('frozen cohort selection changed')
    return manifest


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n' for row in rows), encoding='utf-8')
    temporary.replace(path)


def prepare_resume(output, identity, ids, resume):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists():
        previous = {k:v for k,v in read_json(manifest_path).items() if k != 'status'}
        if not resume or previous != identity:
            raise ValueError('refuse overwrite or changed resume identity')
    elif any(output.iterdir()):
        raise ValueError('unknown output artifacts without identity')
    saved = read_jsonl(output/'traces.jsonl') if (output/'traces.jsonl').exists() else []
    completed_ids = [row.get('source_id') for row in saved]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError('duplicate saved sample IDs')
    if not set(completed_ids) <= set(ids):
        raise ValueError('saved sample ID outside frozen cohort')
    if not manifest_path.exists():
        write_json(manifest_path, identity)
    return saved


def candidate_history(row, registry):
    history = []
    for event in row['trace'].get('events', []):
        data = event.get('data', {})
        outputs = data.get('data', {})
        if event.get('kind') != 'observation' or data.get('status') != 'success' or 'standard_candidates' not in outputs:
            continue
        candidates = outputs['standard_candidates']
        if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
            raise ValueError('invalid candidate observation shape')
        ids = [c.get('category') or c.get('category_id') for c in candidates]
        if len(ids) != 5 or len(set(ids)) != 5 or not set(ids) <= set(registry):
            raise ValueError('candidate observation is not five distinct frozen Registry IDs')
        history.append(ids)
    return history


def _macro_f1(golds, predictions, labels):
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g,p in zip(golds, predictions))
        fp = sum(g != label and p == label for g,p in zip(golds, predictions))
        fn = sum(g == label and p != label for g,p in zip(golds, predictions))
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return sum(scores)/len(scores) if scores else 0.0


def summarize(rows, registry, supported_classes):
    n = len(rows)
    if len({r['source_id'] for r in rows}) != n:
        raise ValueError('duplicate sample ID in scoring')
    totals = Counter()
    skills = Counter()
    successes = Counter()
    failures = Counter()
    interventions = Counter()
    stop_reasons = Counter()
    golds, predictions = [], []
    elapsed = 0.0
    new_sample_count = 0
    for row in rows:
        trace = row['trace']
        final = trace.get('final_result') or {}
        terminated = bool(trace.get('terminated'))
        prediction = final.get('category') if terminated else None
        predicted_level = final.get('grading_level') if terminated else None
        gold, level = row['ground_truth'], row['ground_truth_level']
        if terminated and (prediction not in registry or registry[prediction].get('grading_level') != predicted_level):
            raise ValueError('terminal result disagrees with frozen Registry')
        golds.append(gold)
        predictions.append(prediction)
        correct = terminated and prediction == gold
        level_correct = terminated and predicted_level == level
        totals['terminated'] += terminated
        totals['category_correct'] += correct
        totals['level_correct'] += level_correct
        totals['joint_correct'] += correct and level_correct
        stop_reasons[trace.get('stop_reason', 'unknown')] += 1
        recovered = False
        used_new = False
        for event in trace.get('events', []):
            data = event.get('data', {})
            kind = event.get('kind')
            if kind == 'controller_intervention':
                reason = data.get('reason', 'unknown')
                interventions[reason] += 1
                recovered |= reason in {'forced_finalization', 'invalid_action_finalization_recovery'}
            if kind == 'policy_action':
                tool = data.get('call', data).get('tool')
                if tool:
                    skills[tool] += 1
                    totals['skill_calls'] += 1
                    used_new |= tool not in P14_SKILLS
            if kind == 'observation':
                tool = data.get('skill', 'unknown')
                if data.get('status') == 'success':
                    successes[tool] += 1
                    totals['successful_skill_calls'] += 1
                else:
                    failures[tool] += 1
                    totals['observation_failures'] += 1
                    totals['unavailable_skill_calls'] += data.get('status') == 'unavailable'
            if kind == 'retry':
                totals['retry_events'] += 1
            if kind in {'retry', 'rejected_action'} and data.get('reason') in {'invalid_action','uncallable_skill','invalid_arguments','stop_without_final_result'}:
                totals['invalid_policy_attempts'] += 1
                if data.get('reason') == 'stop_without_final_result':
                    totals['early_stop_attempts'] += 1
            if kind == 'policy_action' and data.get('messages') and not data.get('forced_finalize'):
                totals['accepted_policy_actions'] += 1
        totals['recovered_terminated'] += terminated and recovered
        totals['normal_terminated'] += terminated and not recovered
        new_sample_count += used_new
        history = candidate_history(row, registry)
        totals['top5_first_hit'] += bool(history and gold in history[0])
        totals['top5_last_hit'] += bool(history and gold in history[-1])
        totals['top5_any_hit_diagnostic'] += any(gold in c for c in history)
        totals['samples_with_retrieval'] += bool(history)
        totals['multiple_retrieval_samples'] += len(history) > 1
        if not correct:
            if not history:
                totals['no_retrieval_failures'] += 1
            elif gold not in history[-1]:
                totals['last_top5_miss_failures'] += 1
            elif not terminated:
                totals['hit_but_nonterminal_failures'] += 1
            else:
                totals['candidate_inside_wrong_choice'] += 1
        totals['runner_errors'] += bool(row.get('error'))
        for call in row.get('generation_calls', []):
            meta = call.get('generation', {})
            totals['model_calls'] += 1
            totals['backend_failed_calls'] += bool(call.get('error'))
            totals['total_tokens'] += int(meta.get('token_usage', {}).get('total_tokens') or 0)
            elapsed += float(meta.get('elapsed_seconds') or 0)
    expected_counters = ('terminated','category_correct','level_correct','joint_correct',
                         'normal_terminated','recovered_terminated','top5_first_hit','top5_last_hit',
                         'top5_any_hit_diagnostic','model_calls','total_tokens','skill_calls',
                         'successful_skill_calls','runner_errors','backend_failed_calls',
                         'no_retrieval_failures','last_top5_miss_failures','hit_but_nonterminal_failures',
                         'candidate_inside_wrong_choice','observation_failures','retry_events',
                         'early_stop_attempts','samples_with_retrieval','multiple_retrieval_samples')
    result = {'denominator': n, **{key: totals[key] for key in expected_counters}, **dict(totals),
              'model_elapsed_seconds': elapsed,
              'macro_f1_supported': _macro_f1(golds, predictions, supported_classes),
              'macro_f1_fixed12': _macro_f1(golds, predictions, sorted(registry)),
              'supported_class_count': len(supported_classes), 'fixed_class_count': len(registry),
              'distinct_skills_used': len(skills), 'skills_called': dict(sorted(skills.items())),
              'skills_succeeded': dict(sorted(successes.items())), 'skills_failed': dict(sorted(failures.items())),
              'controller_interventions': dict(sorted(interventions.items())),
              'stop_reasons': dict(sorted(stop_reasons.items()))}
    result.update({'new_skills_called':{k:v for k,v in sorted(skills.items()) if k not in P14_SKILLS},
                   'samples_using_new_skills':new_sample_count,
                   'unavailable_skill_calls':totals['unavailable_skill_calls'],
                   'invalid_policy_attempts':totals['invalid_policy_attempts'],
                   'accepted_policy_actions':totals['accepted_policy_actions'],
                   'legal_action_rate':totals['accepted_policy_actions']/(totals['accepted_policy_actions']+totals['invalid_policy_attempts'])
                       if totals['accepted_policy_actions']+totals['invalid_policy_attempts'] else 0.0})
    for key in ('terminated','category_correct','level_correct','joint_correct','top5_first_hit','top5_last_hit'):
        result[key+'_rate'] = result[key]/n if n else 0.0
    return result
