"""Evaluator-only label access; missing/failed sources stay in the full denominator."""
import argparse
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from artifacts import fingerprint, load_study, put, read, sha
from runner import read_receipts

SEARCH = {'SELECT_CATEGORY_TOPK', 'SEARCH_STANDARD', 'SEARCH_STANDARD_BY_FIELD',
          'SEARCH_STANDARD_BY_CONTEXT', 'SEARCH_STANDARD_BY_TABLE', 'SEARCH_STANDARD_BY_DATABASE', 'RESEARCH_STANDARD'}


def macro(golds, preds, labels):
    values = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(golds, preds))
        fp = sum(g != label and p == label for g, p in zip(golds, preds))
        fn = sum(g == label and p != label for g, p in zip(golds, preds))
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return sum(values) / len(values) if values else 0.


def score_traces(ids, traces, answers, registry):
    if not ids or len(set(ids)) != len(ids) or set(ids) != set(answers) or not set(traces) <= set(ids):
        raise ValueError('score IDs differ')
    counts = Counter()
    details = {}
    golds, preds = [], []
    for ident in ids:
        trace = traces.get(ident) or {}
        final = trace.get('final_result') or {}
        final = final if isinstance(final, Mapping) else {}
        key = final.get('category')
        entry = registry.get(key) if isinstance(key, str) else None
        legal = bool(trace.get('terminated') is True and trace.get('stop_reason') == 'validated_terminal'
                     and entry and final.get('grading_level') == entry['grading_level'])
        gold = answers[ident]
        category = bool(legal and entry['path'] == gold['path'])
        grade = bool(legal and final['grading_level'] == gold['grading_level'])
        history = []
        for program in trace.get('programs', []):
            for step in program.get('steps', []):
                candidates = (step.get('data') or {}).get('standard_candidates')
                if step.get('call', {}).get('tool') in SEARCH and step.get('status') == 'success' and isinstance(candidates, list):
                    keys = [r.get('category') or r.get('category_id') for r in candidates if isinstance(r, dict)]
                    if len(keys) == 5 and len(set(keys)) == 5 and set(keys) <= set(registry):
                        history.append(keys)
        hits = [any(registry[k]['path'] == gold['path'] for k in group) for group in history]
        entry_hits = [bool(set(group) & set(gold['compatible_standard_ids'])) for group in history]
        row = {'terminated': legal, 'category_correct': category, 'grading_correct': grade,
               'joint_correct': category and grade, 'top5_first_hit': bool(hits and hits[0]),
               'top5_final_hit': bool(hits and hits[-1]), 'top5_any_hit': any(hits),
               'top5_compatible_entry_any_hit': any(entry_hits)}
        counts.update({k: int(v) for k, v in row.items()})
        details[ident] = row
        golds.append(tuple(gold['path']))
        preds.append(tuple(entry['path']) if legal else None)
    n = len(ids)
    return {'denominator': n, 'missing_or_failed_traces': n - len(traces),
            **{key: counts[key] for key in ('terminated', 'category_correct', 'grading_correct', 'joint_correct',
                                           'top5_first_hit', 'top5_final_hit', 'top5_any_hit', 'top5_compatible_entry_any_hit')},
            'category_accuracy': counts['category_correct'] / n, 'grading_accuracy': counts['grading_correct'] / n,
            'joint_accuracy': counts['joint_correct'] / n, 'termination_rate': counts['terminated'] / n,
            'macro_f1_supported': macro(golds, preds, set(golds)),
            'macro_f1_all_leaf_paths': macro(golds, preds, {tuple(e['path']) for e in registry.values()}),
            'supported_leaf_paths': len(set(golds)), 'registry_leaf_paths': len({tuple(e['path']) for e in registry.values()}),
            'per_id': details}


def score_run(package, output):
    study, manifest = load_study(package)
    if sha(Path(package) / 'evaluation/gold.json') != manifest['gold_sha256']:
        raise ValueError('gold identity changed')
    identity = read(Path(output) / 'run.json')
    if identity['study_sha256'] != manifest['study_sha256'] or identity['code_sha256'] != fingerprint(manifest['code']) or identity['expected_ids'] != study['expected_ids']:
        raise ValueError('scoring run identity differs')
    receipts = read_receipts(output, study['expected_ids'])
    traces = {}
    for n, ident in enumerate(study['expected_ids'], 1):
        if ident in receipts and receipts[ident]['status'] == 'completed':
            traces[ident] = read(Path(output) / 'sources' / f'{n:04d}' / 'trace.json')
    result = score_traces(study['expected_ids'], traces, read(Path(package) / 'evaluation/gold.json')['answers'], study['registry'])
    result.update(backend=identity['backend'], quality_claim=identity['backend'] in ('local_official_base', 'hosted_qwen35_9b_api'),
                  complete=len(receipts) == len(study['expected_ids']),
                  scope='development validation; grading derives from the chosen standard entry',
                  run_manifest_sha256=sha(Path(output) / 'run.json'))
    if 'api_service' in identity:
        result['api_service'] = identity['api_service']
        result['comparison_scope'] = 'hosted API baseline; not verified equivalent to pinned local FP16 weights'
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = score_run(a.package, a.run)
    put(a.output, result)
    import json
    print(json.dumps({k: v for k, v in result.items() if k != 'per_id'}))
